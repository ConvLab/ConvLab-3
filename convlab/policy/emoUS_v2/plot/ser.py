import json
import os
from glob import glob
from convlab.policy.emoUS_v2.plot.plot_average_sentiment import get_ser
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file", '-f', type=str)
    parser.add_argument("--task-map", '-t', type=str)
    return parser.parse_args()


def single_plot(x, y, label, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    mean = np.array(y["mean"])
    std = np.array(y["std"])

    ax.plot(x, mean)
    ax.fill_between(x,
                    mean+std,
                    mean-std, alpha=0.5)

    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{label}.pdf"))


def get_exp_data(exp_folder):
    data = {"x": [],
            "missing": {"mean": [], "std": []},
            "hallucinate": {"mean": [], "std": []},
            "SER": {"mean": [], "std": []}}
    temp = {}
    for exp in ["experiments", "finished_experiments"]:
        for f in sorted(glob(os.path.join(exp_folder, exp, "*"))):
            temp[f] = {"x": [], "missing": [], "hallucinate": [], "SER": []}
            for i, c in enumerate(sorted(glob(os.path.join(f, "logs", "conversation", "*")))):
                r = get_ser(json.load(open(c))["conversation"])
                temp[f]["x"].append(i)
                temp[f]["missing"].append(r["missing"])
                temp[f]["hallucinate"].append(r["hallucinate"])
                temp[f]["SER"].append(r["SER"])
            print(f, temp[f]["x"])
    for f in temp:
        if len(temp[f]["x"]) > len(data["x"]):
            data["x"] = temp[f]["x"]
    for x in data["x"]:
        for m in ["missing", "hallucinate", "SER"]:
            d = [temp[f][m][x] for f in temp if x < len(temp[f][m])]
            mean = np.mean(d)
            std = np.std(d, ddof=1) / np.sqrt(len(d))
            data[m]["mean"].append(mean)
            data[m]["std"].append(std)
    return data


def plot(data, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for m in ["missing", "hallucinate", "SER"]:
        fig, ax = plt.subplots(figsize=(12, 8))
        for d in data:
            x = d["data"]["x"]
            mean = np.array(d["data"][m]["mean"])
            std = np.array(d["data"][m]["std"])
            # x = np.array(range(mean.shape[0]))
            marker = d.get("marker", "o")
            ax.plot(x,
                    mean,
                    marker=marker,
                    linestyle='--',
                    color=d["color"],
                    label=d["label"])
            ax.fill_between(x,
                            mean+std,
                            mean-std,
                            color=d["color"],
                            alpha=0.5)

        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel(m)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{m}.pdf"))


def main():
    args = arg_parser()
    if args.task_map:
        tasks = json.load(open(args.task_map))
        data = []
        for model in tasks["models"]:
            data.append({
                "label": model["label"],
                "color": tasks["colors"][model["color"]],
                "data": get_exp_data(model["folder"]),
                "marker": model["marker"]})
        plot(data, tasks["result_dir"])

    else:
        data = get_exp_data(args.file)

        for m in ["missing", "hallucinate", "SER"]:
            single_plot(data["x"], data[m], m,
                        os.path.join(args.file, "plot"))


if __name__ == '__main__':
    main()
