from argparse import ArgumentParser
import json
import os
import pandas as pd

from convlab.policy.emoUS_v2.plot.success_all_fail import get_turn_emotion
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
    parser.add_argument("--pick", type=str, default="all")
    parser.add_argument("--result-dir", type=str,
                        default="convlab/policy/emoUS_v2/result")
    return parser.parse_args()


def plot(data, max_turn, result_dir, p, pick="all"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig, ax = plt.subplots(figsize=(6, 6))

    for d_type, para in p.items():
        if pick != "all" and "no-conduct" in d_type:
            continue

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
                        color=p["color"], alpha=0.1)

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

    p_list = ['rl-scratch-conduct',
              'rl-scratch-no-conduct',
              'rl-pretrain-conduct',
              'rl-pretrain-no-conduct',
              'sl-conduct',
              'sl-no-conduct']
    c1 = "cornflowerblue"
    c2 = "mediumseagreen"
    c3 = "coral"
    p = {
        p_list[0]: {pick: {"color": c1, "label": p_list[0], "marker": "o"}},
        p_list[1]: {pick: {"color": c1, "label": p_list[1], "marker": "x"}},
        p_list[2]: {pick: {"color": c2, "label": p_list[2], "marker": "o"}},
        p_list[3]: {pick: {"color": c2, "label": p_list[3], "marker": "x"}},
        p_list[4]: {pick: {"color": c3, "label": p_list[4], "marker": "o"}},
        p_list[5]: {pick: {"color": c3, "label": p_list[5], "marker": "x"}}
    }
    p_list = [x for x in p]

    for index, folder in enumerate([args.file1, args.file2, args.file3, args.file4, args.file5, args.file6]):

        data[p_list[index]], max_turn = get_turn_emotion(
            json.load(open(os.path.join(folder, "conversation.json")))["conversation"])

    if pick == "all":

        basic_info = {}
        col = ["Complete", "Success", "Success strict", "turns"]
        for index, folder in enumerate([args.file1, args.file2, args.file3, args.file4, args.file5, args.file6]):
            basic_info[p_list[index]] = [json.load(
                open(os.path.join(folder, "conversation_result.json")))["basic_info"][s] for s in col]

        df = pd.DataFrame(basic_info, index=col)
        df.to_csv(os.path.join(args.result_dir, "basic_info.csv"))

    plot(data,
         max_turn,
         args.result_dir,
         p,
         pick)


if __name__ == "__main__":
    main()
