import argparse
import json
from eval_generic import EvalHandler, compute_metric
import time

import os
import pandas as pd

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", required=True)
    args = parser.parse_args()
    model_type = args.model_type
    model_name = args.model_name

    data_dir = "data_mmlu"

    handler = EvalHandler(model_name, model_type)
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
