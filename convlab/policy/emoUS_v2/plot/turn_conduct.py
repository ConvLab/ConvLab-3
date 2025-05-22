import json
from argparse import ArgumentParser
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file", "-f", type=str)
    parser.add_argument("--result-dir", "-r", type=str,
                        default="convlab/policy/emoUS_v2/results/turn_conduct/")

    return parser.parse_args()


def get_turn_conduct_distribution(conversation, pick="all", dataset=False):

    conduct = {"neutral": [0]*20,
               "compassionate": [0]*20,
               "apologetic": [0]*20,
               "enthusiastic": [0]*20,
               "appreciative": [0]*20}

    if dataset:
        folder = "data/unified_datasets/emowoz/data_copy/"
        conduct_map = json.load(
            open(os.path.join(folder, "system_conduct_label.json")))
        data = json.load(open(os.path.join(folder, "system_conduct.json")))
        for turn_id, num in data.items():
            index = int(turn_id.split("-")[-1])//2
            if index > 19:
                continue
            c = conduct_map[str(num)]
            print(c, index)
            conduct[c][index] += 1
        return conduct, None
    turn = Counter([len(dialog["log"])//2 for dialog in conversation])
    normalize = [0]*20
    for dialog in conversation:
        if pick == "Success strict" and dialog["Success strict"] != 1:
            continue
        if pick == "Not Success strict" and dialog["Success strict"] == 1:
            continue

        for t, turn in enumerate(dialog["log"]):
            if turn["role"] == "usr":
                continue
            if int(t/2) >= 20:
                continue
            conduct[turn["conduct"]][int(t / 2)] += 1
            normalize[int(t / 2)] += 1

    conduct = {k: np.array(v) for k, v in conduct.items()}
    return conduct, np.array(normalize)


def plot(data, max_turn, result_dir, normalize=None, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    width = 0.5
    x = list(range(max_turn))
    bottom = np.zeros(max_turn)
    num = np.sum([data[x][0] for x in data])
    print(num, "num")

    for c, d in data.items():
        if normalize is not None:
            d = d / normalize
        d = d / num
        print(d, "d")
        ax.bar(x,
               d,
               width,
               label=c,
               # color=colors[c],
               bottom=bottom)
        bottom += d

    ax.legend()
    ax.set_xlabel("turn")
    ax.set_ylabel("% of dialogues")
    # # ax.set_ylim([-1.0, 0.4])
    ax.set_xticks([t for t in range(0, max_turn, 1)])
    # plt.grid(axis='x', color='0.95')
    # plt.grid(axis='y', color='0.95')
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{pick}-conduct.pdf"))


def main():
    args = arg_parser()
    file_name = args.file
    # data, normalize = get_turn_conduct_distribution(
    #     json.load(open(file_name))["conversation"], "all",)
    data, normalize = get_turn_conduct_distribution(
        None, "all", True)
    # if pick == "all":
    #     normalize = None
    plot(data,
         20,
         args.result_dir,)
    # normalize)


if __name__ == "__main__":
    main()
