import argparse
import json
from eval_generic import EvalHandler, compute_metric
import time

import os
import pandas as pd

TASKS = ["th"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    model_name = args.model_name

    data_dir = "data_xcopa"

    handler = EvalHandler(
        model_name, args.model_type, max_length=args.max_length, n_choice=2
    )
    run_results = {}
    start_time = time.time()
    for task in TASKS:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", task + "_dev.csv"), header=None
        )[:5]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", task + "_test.csv"), header=None
        )
        resp = handler.evaluate(dev_df, test_df, task)
        run_results[task] = resp

    model_name_escape = model_name.replace("/", "-")
    output_filename = f"run_results_{model_name_escape}.json"
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
