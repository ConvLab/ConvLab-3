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


def training_info(conversation: dict):
    for seed, dialog in conversation.items():
        if "All_user_sim" in dialog["info"]:
            # old version
            return {"complete": np.average(dialog["info"]["All_user_sim"]),
                    "task_succ": np.average(dialog["info"]["All_evaluator"]),
                    "task_succ_strict": np.average(dialog["info"]["All_evaluator_strict"])}


def main():
    args = arg_parser()
    folder = os.path.join(args.folder, "logs", "conversation")
    files = sorted(glob(os.path.join(folder, "*.json")))
    results = {}
    for i, file in enumerate(files):
        results[i] = training_info(json.load(open(file)))
    pprint(results)


if __name__ == "__main__":
    main()
