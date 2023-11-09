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
    emotion_system_label = json.load(
        open("data/unified_datasets/emowoz/data/emotion_system_label.json"))
    args = arg_parser()
    for golden_conduct, c1 in tqdm(emotion_system_label.items()):
        for target_conduct, c2 in emotion_system_label.items():
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
                        id_list[x]["ori_emo"] = d["gen_emotion"]
                if d["id"] == id_list[x]["pred"]:
                    id_list[x]["new_emo"] = d["gen_emotion"]
            if c1 == c2:
                for d in ori_file["dialog"]:
                    x = '-'.join(d["id"].split("-")[:2])
                    if x not in id_list:
                        continue
                    if d["id"] == x:
                        id_list[x]["ori_emo"] = d["gen_emotion"]
            label = []
            pred = []
            for x, r in id_list.items():
                if "ori_emo" not in r or "new_emo" not in r:
                    continue
                label.append(r["ori_emo"])
                pred.append(r["new_emo"])

            normalize = None
            dirname = "figs"
            if args.normalize:
                f_name += "-normalize"
                dirname = os.path.join(dirname, "normalize")
                normalize = "true"
            emotion_plot(label,
                         pred,
                         dirname=dirname,
                         file_name=f_name,
                         normalize=normalize)


if __name__ == "__main__":
    main()
