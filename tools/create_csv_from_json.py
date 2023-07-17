import json
import pandas as pd

if __name__ == "__main__":
    labels = ["A", "B"]
    subset = "test"

    with open(f"/home/kunato/llm-evaluation/data_xcopa/{subset}.th.jsonl") as f:
        lines = f.readlines()
        results = []
        for l in lines:
            row = json.loads(l)
            premise = row["premise"]
            choice1 = row["choice1"]
            choice2 = row["choice2"]
            label_idx = row["label"]
            label = labels[label_idx]
            results.append([premise, choice1, choice2, label])

    df = pd.DataFrame(results)
    df.to_csv(f"{subset}.csv", header=False, index=False)
