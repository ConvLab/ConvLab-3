import json
from argparse import ArgumentParser
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from datasets import load_metric
from convlab.policy.genTUS.golden_nlg_evaluation import ser_v2


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--evaluation-type", type=str,
                        help="emotion, utterance")
    parser.add_argument("--result-dir", type=str,
                        help="dir to save evaluation results")
    return parser.parse_args()


def emotion_score(golden_emotions, gen_emotions, dirname=".", no_neutral=False):
    labels = ["Neutral", "Fearful", "Dissatisfied",
              "Apologetic", "Abusive", "Excited", "Satisfied"]
    if no_neutral:
        labels = labels[1:]

    macro_f1 = metrics.f1_score(golden_emotions, gen_emotions, average="macro")
    sep_f1 = metrics.f1_score(
        golden_emotions, gen_emotions, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_emotions, gen_emotions, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"emotion.png"))
    r = {"label": labels,
         "macro_f1": float(macro_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    return r


def sentiment_score(golden_sentiment, gen_sentiment, dirname="."):
    labels = ["Neutral", "Negative", "Positive"]

    macro_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average="macro")
    sep_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_sentiment, gen_sentiment, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"sentiment.png"))
    r = {"label": labels,
         "macro_f1": float(macro_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    return r


def nlg_scores(file_name, dirname=""):
    bleu_metric = load_metric("sacrebleu")
    if not dirname:
        dirname = os.path.dirname(file_name)
    with open(file_name, "r") as fin:
        data = json.load(fin)
    labels = [[x["label"]] for x in data]
    gen_utts = [x["predict"] for x in data]
    gen_acts = [get_semantic_action(x["in"]) for x in data]
    print(gen_acts[0])

    bleu_score = bleu_metric.compute(predictions=gen_utts,
                                     references=labels,
                                     force=True)
    print(bleu_score)
    ser = ser_v2(gen_acts, gen_utts)
    print(ser)


def emotion_scores(file_name, dirname=""):
    if not dirname:
        dirname = os.path.dirname(file_name)
    with open(file_name, "r") as fin:
        data = json.load(fin)
    golden_emotions = [x["label"] for x in data]
    gen_emotions = [x["predict"] for x in data]
    r = emotion_score(golden_emotions, gen_emotions, dirname)
    print(r)


def get_semantic_action(text):
    text = text.split("Your action is: ")[-1]
    text = text.split("\n")[0]
    action = json.loads(text)
    return action


def main():
    args = arg_parser()
    if args.evaluation_type == "emotion":
        emotion_scores(args.data, args.result_dir)
    elif args.evaluation_type == "nlg":
        nlg_scores(args.data, args.result_dir)


if __name__ == "__main__":
    main()
