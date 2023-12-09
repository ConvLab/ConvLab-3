from glob import glob
import json
from argparse import ArgumentParser
import os
import numpy as np
from pprint import pprint


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="")

    return parser.parse_args()


def training_info(conversation):
    r = {"complete": [], "task_succ": [], "task_succ_strict": []}
    for dialog in conversation:
        r["complete"].append(dialog["Complete"])
        r["task_succ"].append(dialog["Success"])
        r["task_succ_strict"].append(dialog["Success strict"])

    return {"complete": np.average(r["complete"]),
            "task_succ": np.average(r["task_succ"]),
            "task_succ_strict": np.average(r["task_succ_strict"])}


def _training_info(conversation: dict):
    r = {"complete": [], "task_succ": [], "task_succ_strict": []}
    for seed, dialog in conversation.items():
        if "info" in dialog:
            if "All_user_sim" in dialog["info"]:
                # old version
                return {"complete": np.average(dialog["info"]["All_user_sim"]),
                        "task_succ": np.average(dialog["info"]["All_evaluator"]),
                        "task_succ_strict": np.average(dialog["info"]["All_evaluator_strict"])}
            if "Complete" in dialog["info"]:
                # new version
                r["complete"].append(dialog["info"]["Complete"])
                r["task_succ"].append(dialog["info"]["Success"])
                r["task_succ_strict"].append(dialog["info"]["Success_strict"])
        if "Complete" in dialog:
            # new version
            r["complete"].append(dialog["Complete"])
            r["task_succ"].append(dialog["Success"])
            r["task_succ_strict"].append(dialog["Success strict"])

    return {"complete": np.average(r["complete"]),
            "task_succ": np.average(r["task_succ"]),
            "task_succ_strict": np.average(r["task_succ_strict"])}


def main():
    args = arg_parser()
    folder = os.path.join(args.folder, "logs", "conversation")
    files = sorted(glob(os.path.join(folder, "*.json")))
    results = {}
    for i, file in enumerate(files):
        conversation = json.load(open(file))
        results[i] = training_info(conversation["conversation"])
    pprint(results)


if __name__ == "__main__":
    main()
