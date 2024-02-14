from argparse import ArgumentParser
import json
import os
import pandas as pd

from convlab.policy.emoUS_v2.plot.success_all_fail import get_turn_emotion
import matplotlib.pyplot as plt
from convlab.nlg.evaluate import fine_SER
from convlab.nlg.evaluate_unified_datasets_v2 import ser_new


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--task-map", '-t', type=str,
                        help="sl-conduct")
    parser.add_argument("--pick", type=str, default="all")
    return parser.parse_args()


def plot(data, max_turn, result_dir, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 3))

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


def get_ser(conversation):
    acts = []
    utts = []
    for d in conversation:
        for t in d["log"]:
            if t["role"] == "sys":
                acts.append(t["act"])
                utts.append(t["utt"])
    # shutong: new code here
    # missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
    #     acts, utts)
    missing, hallucinate, total, hallucination_dialogs, missing_dialogs = ser_new(
        acts, utts)
    return {"missing": missing/total, "hallucinate": hallucinate/total, "SER": (missing+hallucinate)/total}


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
    ner = {}
    ner_col = ["missing", "hallucinate", "SER"]
    for model, config in task_map["models"].items():
        conversation = json.load(open(config["file"]))
        config["data"], max_turn = get_turn_emotion(
            conversation["conversation"])
        config["color"] = colors[config["color"]]
        data[model] = config
        # ner[model] = get_ser(conversation["conversation"])

    plot(data,
         max_turn,
         result_dir,
         pick)

    # if pick == "all":
    #     basic_info = {}
    #     for model, config in task_map["models"].items():
    #         folder = os.path.dirname(config["file"])
    #         basic_info[model] = [json.load(
    #             open(os.path.join(folder, "conversation_result.json")))["basic_info"][s] for s in col]
    #         basic_info[model].extend([ner[model][s] for s in ner_col])

    #     df = pd.DataFrame(basic_info, index=col+ner_col).T
    #     df.to_csv(os.path.join(result_dir, "basic_info.csv"))


if __name__ == "__main__":
    main()
