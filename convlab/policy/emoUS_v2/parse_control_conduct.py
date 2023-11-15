import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tqdm import tqdm


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--generate-file", '-g', type=str, default="")
    parser.add_argument("--input-file", '-i', type=str, default="")
    parser.add_argument("--original-file", '-o', type=str, default="")
    parser.add_argument("--normalize", '-n', action="store_true")
    parser.add_argument("--result-dir", '-r', type=str, default="figs")
    return parser.parse_args()


def emotion_plot(golden_emotions, gen_emotions, dirname=".", file_name="emotion", no_neutral=False, normalize=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    labels = ["Neutral", "Fearful", "Dissatisfied",
              "Apologetic", "Abusive", "Excited", "Satisfied"]
    if no_neutral:
        labels = labels[1:]
    cm = metrics.confusion_matrix(
        golden_emotions, gen_emotions, normalize=normalize, labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    plt.xticks(rotation=45)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, f"{file_name}.pdf"))
    with open(os.path.join(dirname, f"{file_name}.txt"), "w") as f:
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                if i == j:
                    continue
                if cm[i][j] < 0.01:
                    continue
                f.write(f"{labels[i]} [{cm[i][j]:.2f}] {labels[j]}' \n")


def main():
    system_conduct_label = json.load(
        open("data/unified_datasets/emowoz/data/system_conduct_label.json"))
    args = arg_parser()
    for golden_conduct, c1 in tqdm(system_conduct_label.items()):
        for target_conduct, c2 in system_conduct_label.items():
            f_name = f"{c1}-{c2}"
            golden_conduct = int(golden_conduct)
            target_conduct = int(target_conduct)

            input_file = pd.read_csv(args.input_file)
            gen_file = json.load(open(args.generate_file))
            ori_file = json.load(open(args.original_file))
            id_list = {}

            for _, l in input_file.iterrows():
                # neural -> enthusiastic
                x = f'{l["dialogue_id"]}-{l["turn_id"]+1}'

                if l["gold_conduct"] == golden_conduct and l["target_conduct"] == golden_conduct:
                    if x not in id_list:
                        id_list[x] = {}
                    id_list[x]["label"] = f'{x}-{l["target_conduct"]}'

                if l["gold_conduct"] == golden_conduct and l["target_conduct"] == target_conduct:
                    if x not in id_list:
                        id_list[x] = {}
                    id_list[x]["pred"] = f'{x}-{l["target_conduct"]}'

            for d in gen_file["dialog"]:
                x = '-'.join(d["id"].split("-")[:2])
                if x not in id_list:
                    continue
                if c1 != c2:
                    if d["id"] == id_list[x]["label"]:
                        temp = parser_system_emotion_user(d, "ori")
                        for k, v in temp.items():
                            id_list[x][k] = v
                if d["id"] == id_list[x]["pred"]:
                    temp = parser_system_emotion_user(d, "new")
                    for k, v in temp.items():
                        id_list[x][k] = v
            if c1 == c2:
                for d in ori_file["dialog"]:
                    x = '-'.join(d["id"].split("-")[:2])
                    if x not in id_list:
                        continue
                    if d["id"] == x:
                        temp = parser_system_emotion_user(d, "ori")
                        for k, v in temp.items():
                            id_list[x][k] = v

            label = []
            pred = []
            for x, r in id_list.items():
                if "ori_emotion" not in r or "new_emotion" not in r:
                    continue
                label.append(r["ori_emotion"])
                pred.append(r["new_emotion"])

            normalize = None
            dirname = args.result_dir
            if args.normalize:
                f_name += "-normalize"
                dirname = os.path.join(dirname, "normalize")
                normalize = "true"
            emotion_plot(label,
                         pred,
                         dirname=dirname,
                         file_name=f_name,
                         normalize=normalize)
            json.dump(id_list, open(os.path.join(
                dirname, f"{f_name}.json"), "w"))


def parser_system_emotion_user(d, x: str):
    if x not in ["new", "ori"]:
        raise ValueError("x should be new or ori")

    return {f"{x}_system": json.loads(d["input"])["system"],
            f"{x}_emotion": d["gen_emotion"],
            f"{x}_utterance": d["gen_utts"]}


if __name__ == "__main__":
    main()
