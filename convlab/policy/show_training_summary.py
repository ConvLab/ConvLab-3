from glob import glob
import json
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from pprint import pprint


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="")
    parser.add_argument("--task-map", "-t", type=str, default="")
    parser.add_argument("--plot", "-p", action="store_true")

    return parser.parse_args()


def training_info(conversation):
    r = {"complete": [],
         "task_succ": [],
         "task_succ_strict": [],
         "sentiment": []}
    for dialog in conversation:
        r["complete"].append(dialog["Complete"])
        r["task_succ"].append(dialog["Success"])
        r["task_succ_strict"].append(dialog["Success strict"])
        # sentiment = []
        for turn in dialog["log"]:
            if turn["role"] == "usr":
                # sentiment.append(get_sentiment(turn["emotion"]))
                # r["sentiment"].append(np.sum(sentiment))
                r["sentiment"].append(get_sentiment(turn["emotion"]))
    return r


def get_sentiment(emotion: str):
    emotion = emotion.lower()
    if emotion in ["dissatisfied", "abusive"]:
        return -1  # -1
    if emotion in ["satisfied"]:
        return 1  # 1
    return 0  # 0


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


def plot(data: dict, folder: str, title: str = None):
    import matplotlib.pyplot as plt

    if not os.path.exists(folder):
        os.makedirs(folder)
    for m in ["complete", "task_succ", "task_succ_strict", "sentiment"]:
        fig, ax = plt.subplots()
        for label, exp in data.items():
            d = exp["result"]
            x = np.array(d['x'])*1000
            mean = np.array(d[m]["mean"])
            std = np.array(d[m]["std"])
            marker = d.get("marker", "o")
            ax.plot(x,
                    mean,
                    marker=marker,
                    linestyle='--',
                    color=exp["color"],
                    label=label)
            ax.fill_between(x,
                            mean+std,
                            mean-std,
                            color=exp["color"],
                            alpha=0.1)

        # workaround: baseline model for EmoTOD
        # if m == "sentiment":
        #     mean = np.array([0.3559166155732679]*len(x))
        #     std = np.array([0.007344526665759563]*len(x))
        #     ax.plot(x,
        #             mean,
        #             marker="o",
        #             color="tab:gray",
        #             label="$EmoTOD$")
        #     ax.fill_between(x,
        #                     mean+std,
        #                     mean-std,
        #                     color="tab:gray",
        #                     alpha=0.1)

        ax.legend()
        if title:
            ax.set_title(title)
        ax.set_xlabel("# of dialog")
        ax.set_ylabel(m)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{m}.pdf"))
        # plt.grid(axis='x', color='0.95')
        # plt.grid(axis='y', color='0.95')
        # plt.show()


def merge_seeds(data):
    epochs = {}
    for exp in data:
        for e, info in exp.items():
            if e not in epochs:
                epochs[e] = info
            else:
                for m in info:
                    epochs[e][m] += info[m]
    r = {m: {"mean": [], "std": []}
         for m in ["complete", "task_succ", "task_succ_strict", "sentiment"]}
    r["x"] = sorted(list(epochs.keys()))
    for e in r["x"]:
        for m in epochs[0]:
            r[m]["mean"].append(np.mean(epochs[e][m]))
            r[m]["std"].append(np.std(epochs[e][m], ddof=1) /
                               np.sqrt(len(epochs[e][m])))
    return r


class Table:
    def __init__(self, max_epoch=15):
        self.success = {"exp": [], "seed": []}
        self.sentiment = {"exp": [], "seed": []}
        self.max_epoch = max_epoch
        for e in range(max_epoch+1):
            self.success[str(e)] = []
            self.sentiment[str(e)] = []

    def update(self, folder, data):
        x = folder.split("/")
        exp = x[-3]
        seed = x[-1]
        self.success["exp"].append(exp)
        self.success["seed"].append(seed)
        self.sentiment["exp"].append(exp)
        self.sentiment["seed"].append(seed)
        for e in range(self.max_epoch+1):
            if e < len(data):
                self.success[str(e)].append(
                    np.mean(data[e]["task_succ_strict"]))
                self.sentiment[str(e)].append(np.mean(data[e]["sentiment"]))
            else:
                self.success[str(e)].append(0)
                self.sentiment[str(e)].append(0)

    def to_csv(self, path):
        pd.DataFrame(self.success).to_csv(os.path.join(path, "success.csv"))
        pd.DataFrame(self.sentiment).to_csv(
            os.path.join(path, "sentiment.csv"))


def main():
    args = arg_parser()
    if args.plot:
        table = Table(15)
        task_map = json.load(open(args.task_map))
        results = {}
        colors = task_map["colors"]
        for exp in task_map["models"]:
            print("plot...", exp["folder"])
            folder = exp["folder"]
            data = []
            for experiment in ["experiments", "finished_experiments"]:
                for exp_folder in glob(os.path.join(folder, experiment, "*")):
                    temp = {}
                    path = os.path.join(
                        exp_folder, "logs", "conversation", "*.json")
                    for epoch, file in enumerate(sorted(glob(path))):
                        conversation = json.load(open(file))
                        temp[epoch] = training_info(
                            conversation["conversation"])
                    data.append(temp)
                    print(exp_folder)
                    print([np.mean(temp[x]["task_succ_strict"]) for x in temp])
                    print([np.mean(temp[x]["sentiment"]) for x in temp])
                    table.update(exp_folder, temp)

            r = merge_seeds(data)
            results[exp["label"]] = {
                "result": r, "color": colors[exp["color"]], "marker": exp["marker"]}
        plot(results, task_map["result_dir"])
        table.to_csv(task_map["result_dir"])

    else:
        folder = os.path.join(args.folder, "logs", "conversation")
        files = sorted(glob(os.path.join(folder, "*.json")))
        results = {}
        for i, file in enumerate(files):
            conversation = json.load(open(file))
            r = training_info(conversation["conversation"])
            results[str(i)] = {x: np.average(r[x]) for x in r}
        print(results)


if __name__ == "__main__":
    main()
