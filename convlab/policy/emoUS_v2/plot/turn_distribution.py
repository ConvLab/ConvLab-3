import json
from argparse import ArgumentParser
from collections import Counter
import os
import matplotlib.pyplot as plt


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--file1", type=str,
                        help="sl-conduct")
    parser.add_argument("--file2", type=str,
                        help="sl-no-conduct")
    parser.add_argument("--file3", type=str,
                        help="rl-conduct")
    parser.add_argument("--file4", type=str,
                        help="rl-no-conduct")
    parser.add_argument("--file5", type=str,
                        help="sl-conduct")
    parser.add_argument("--file6", type=str,
                        help="sl-no-conduct")
    parser.add_argument("--result-dir", type=str,
                        default="convlab/policy/emoUS_v2/result")

    return parser.parse_args()


def get_turn_distribution(conversation):
    turn = Counter([len(dialog["log"])//2 for dialog in conversation])
    accumulate_turn = [0]*20
    for k, v in turn.items():
        for i in range(20):
            if i < k:
                accumulate_turn[i] += v
    accumulate_turn = [x/500 for x in accumulate_turn]
    return accumulate_turn


def plot(data, max_turn, result_dir, p, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    x = list(range(max_turn))
    for d_type, para in p.items():

        d = data[d_type]
        p = para[pick]

        ax.plot(x,
                d,
                marker=p["marker"],
                linestyle='--',
                color=p["color"],
                label=p["label"])

    ax.legend()
    ax.set_xlabel("turn")
    ax.set_ylabel("% of dialogues")
    # ax.set_ylim([-1.0, 0.4])
    ax.set_xticks([t for t in range(0, max_turn, 2)])
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{pick}.png"))


def main():
    args = arg_parser()
    p_list = ['rl-scratch-conduct',
              'rl-scratch-no-conduct',
              'rl-pretrain-conduct',
              'rl-pretrain-no-conduct',
              'sl-conduct',
              'sl-no-conduct']
    c1 = "cornflowerblue"
    c2 = "mediumseagreen"
    c3 = "coral"
    pick = "turn"
    p = {
        p_list[0]: {pick: {"color": c1, "label": p_list[0], "marker": "o"}},
        p_list[1]: {pick: {"color": c1, "label": p_list[1], "marker": "x"}},
        p_list[2]: {pick: {"color": c2, "label": p_list[2], "marker": "o"}},
        p_list[3]: {pick: {"color": c2, "label": p_list[3], "marker": "x"}},
        p_list[4]: {pick: {"color": c3, "label": p_list[4], "marker": "o"}},
        p_list[5]: {pick: {"color": c3, "label": p_list[5], "marker": "x"}}
    }
    p_list = [x for x in p]
    data = {}
    for index, folder in enumerate([args.file1, args.file2, args.file3, args.file4, args.file5, args.file6]):

        data[p_list[index]] = get_turn_distribution(
            json.load(open(os.path.join(folder, "conversation.json")))["conversation"])

    plot(data, 20, args.result_dir, p, pick=pick)


if __name__ == "__main__":
    main()
