import json
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def compute_metric(output_filename):
    with open(output_filename, "r") as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]["pred_answers"]
        gold_answers = run_results[task]["gold_answers"]
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold:
                acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


class EvalHandler:
    def __init__(self, pretrained_or_path, model_type) -> None:
        self.choices = ["A", "B", "C", "D"]
        model, tokenizer = self.load(pretrained_or_path, model_type)
        self.model = model
        self.tokenizer = tokenizer

    def load(self, pretrained_or_path, model_type):
        n_gpus = torch.cuda.device_count()

        if model_type == "llama":
            # we use tensor parallel for loading llama
            tokenizer = LlamaTokenizer.from_pretrained(
                pretrained_or_path, use_fast=False, padding_side="left"
            )

            model = LlamaForCausalLM.from_pretrained(
                pretrained_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
            )
            model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

            tokenizer.pad_token_id = (
                0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            )
            tokenizer.bos_token_id = 1
        else:
            # mpt-30b's tokenizer only has the fast version
            use_fast = "mosaicml/mpt-30b" in pretrained_or_path
            # however, tensor parallel for running falcon will occur bugs
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_or_path, use_fast=use_fast, padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.pad_token_id = 0

        model.eval()

        return model, tokenizer

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            self.format_subject(subject)
        )
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def prepare_input(self, prompts):
        input_tokens = self.tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )
        input_tokens = {
            k: input_tokens[k]
            for k in input_tokens
            if k in ["input_ids", "attention_mask"]
        }
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to("cuda")

        return input_tokens

    def batch_infer(self, prompts):
        batch_size = 8
        answers = []
        for batch_input in tqdm(self.batch_split(prompts, batch_size)):
            encode_inputs = self.prepare_input(batch_input)
            outputs = self.model.generate(
                **encode_inputs,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
            answers.extend(
                self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )
        answers = [answer[-1] for answer in answers]
        return answers

    def batch_split(self, prompts, batch_num):
        batch_prompts = []
        mini_batch = []
        for prompt in prompts:
            mini_batch.append(prompt)
            if len(mini_batch) == batch_num:
                batch_prompts.append(mini_batch)
                mini_batch = []
        if len(mini_batch) != 0:
            batch_prompts.append(mini_batch)
        return batch_prompts

    def evaluate(self, dev_df, test_df, task):
        records = []
        for i in range(test_df.shape[0]):
            prompt_end = self.format_example(test_df, i, include_answer=False)
            train_prompt = self.gen_prompt(dev_df, task)
            prompt = train_prompt + prompt_end
            while len(self.tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

        pred_answers = self.batch_infer([record["prompt"] for record in records])
        gold_answers = [record["answer"] for record in records]
        return {"pred_answers": pred_answers, "gold_answers": gold_answers}
