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
    args = parser.parse_args()
    model_type = args.model_type
    model_name = args.model_name

    data_dir = "data_xcopa"

    handler = EvalHandler(model_name, model_type, language="th")
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
    output_filename = "run_results_%s.json" % (model_name)
    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
