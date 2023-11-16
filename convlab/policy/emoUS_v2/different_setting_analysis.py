from argparse import ArgumentParser
import json
import os

from convlab.policy.emoUS_v2.analysis import get_turn_emotion
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
    parser.add_argument("--pick", type=str, default="all")
    parser.add_argument("--result-dir", type=str,
                        default="convlab/policy/emoUS_v2/result")
    return parser.parse_args()


def plot(data, max_turn, result_dir, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))
    p = {
        'sl-conduct': {pick: {"color": "lightcoral", "label": "sl-conduct", "marker": "o"}},
        'sl-no-conduct': {pick: {"color": "lightcoral", "label": "sl-no-conduct", "marker": "x"}},
        'rl-conduct': {pick: {"color": "cornflowerblue", "label": "rl-conduct", "marker": "o"}},
        'rl-no-conduct': {pick: {"color": "cornflowerblue", "label": "rl-no-conduct", "marker": "x"}}
    }
    for d_type, para in p.items():
        d = data[d_type]
        name = pick
        p = para[pick]
        ax.plot(d['x'],
                d[f"{name}_mean"],
                marker=p["marker"],
                linestyle='--',
                color=p["color"],
                label=p["label"])
        ax.fill_between(d['x'],
                        d[f"{name}_mean"]+d[f"{name}_std"],
                        d[f"{name}_mean"]-d[f"{name}_std"],
                        color=p["color"], alpha=0.2)

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


def main():
    args = arg_parser()
    data = {}

    data["sl-conduct"], max_turn = get_turn_emotion(
        json.load(open(args.file1))["conversation"])
    data["sl-no-conduct"], max_turn = get_turn_emotion(
        json.load(open(args.file2))["conversation"])
    data["rl-conduct"], max_turn = get_turn_emotion(
        json.load(open(args.file3))["conversation"])
    data["rl-no-conduct"], max_turn = get_turn_emotion(
        json.load(open(args.file4))["conversation"])
    plot(data,
         max_turn,
         args.result_dir,
         pick=args.pick)


if __name__ == "__main__":
    main()
