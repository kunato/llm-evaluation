# %%
import json

with open("/home/kunato/llm-evaluation/results/xglm_en.json") as f:
    data = json.load(f)
# %%
labels = ["A", "B", "C", "D"]
for k in data.keys():
    pred_answers = data[k]["pred_answers"]
    gold_answers = data[k]["gold_answers"]
    for ans in pred_answers:
        assert ans in labels
# %%
