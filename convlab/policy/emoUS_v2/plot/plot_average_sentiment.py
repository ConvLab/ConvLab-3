from argparse import ArgumentParser
import json
import os
import pandas as pd

from convlab.policy.emoUS_v2.plot.success_all_fail import get_turn_emotion
import matplotlib.pyplot as plt


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--task-map", '-t', type=str,
                        help="sl-conduct")
    parser.add_argument("--pick", type=str, default="all")
    return parser.parse_args()


def plot(data, max_turn, result_dir, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))

    for model, info in data.items():
        d = info["data"]
        ax.plot(d['x'],
                d[f"{pick}_mean"],
                marker=info["marker"],
                linestyle='--',
                color=info["color"],
                label=info["label"])
        ax.fill_between(d['x'],
                        d[f"{pick}_mean"]+d[f"{pick}_std"],
                        d[f"{pick}_mean"]-d[f"{pick}_std"],
                        color=info["color"], alpha=0.1)

    ax.legend()
    ax.set_xlabel("turn")
    ax.set_ylabel("Sentiment")
    # ax.set_ylim([-1.0, 0.4])
    ax.set_xticks([t for t in range(0, max_turn, 2)])
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{pick}.png"))


def success_table():
    pass


def main():
    args = arg_parser()
    data = {}
    pick = args.pick
    if "-" in pick:
        pick = pick.replace("-", " ")

    task_map = json.load(open(args.task_map))
    result_dir = task_map["result_dir"]
    colors = task_map["colors"]

    col = ["Complete", "Success", "Success strict", "turns"]
    basic_info = {}
    data = {}
    for model, config in task_map["models"].items():
        conversation = json.load(open(config["file"]))
        config["data"], max_turn = get_turn_emotion(
            conversation["conversation"])
        config["color"] = colors[config["color"]]
        data[model] = config

    plot(data,
         max_turn,
         result_dir,
         pick)

    if pick == "all":
        basic_info = {}
        col = ["Complete", "Success", "Success strict", "turns"]
        for model, config in task_map["models"].items():
            folder = os.path.dirname(config["file"])
            basic_info[model] = [json.load(
                open(os.path.join(folder, "conversation_result.json")))["basic_info"][s] for s in col]

        df = pd.DataFrame(basic_info, index=col)
        df.to_csv(os.path.join(result_dir, "basic_info.csv"))


if __name__ == "__main__":
    main()
