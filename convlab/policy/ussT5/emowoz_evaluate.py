import json
import os
from argparse import ArgumentParser
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from convlab.policy.tus.unify.util import create_goal, load_experiment_dataset
from convlab.policy.ussT5.evaluate import tri_convert

from datasets import load_metric


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="model name")
    parser.add_argument("--data", default="emowoz+dialmage", type=str)
    parser.add_argument("--gen-file", type=str)
    parser.add_argument("--stop", default=-1, type=int)
    return parser.parse_args()


def build_data(raw_data):
    sentiments = {}
    for sentiment, index in json.load(open("convlab/policy/emoUS/sentiment.json")).items():
        sentiments[int(index)] = sentiment
    data = {"input_text": [], "target_text": []}
    for prefix in ["satisfaction score: ", "action prediction: ", "utterance generation: "]:
        for d in raw_data:
            utt = ""
            turn_len = len(d["turns"])
            for index, turn in enumerate(d["turns"]):
                if turn["speaker"] == "user":
                    if index == turn_len - 2:
                        break
                    if index == 0:
                        utt = prefix + turn["utterance"]
                    else:
                        utt += ' ' + turn["utterance"]
                else:
                    if index == 0:
                        print("this should no happen (index == 0)")
                        utt = prefix + turn["utterance"]
                    if index == turn_len - 1:
                        print("this should no happen (index == turn_len - 1)")
                        continue

                    utt += ' ' + turn["utterance"]

                    data["input_text"].append(utt)
                    if prefix == "satisfaction score: ":
                        data["target_text"].append(
                            sentiments[d["turns"][index+1]["emotion"][-1]["sentiment"]])
                    elif prefix == "action prediction: ":
                        data["target_text"].append(
                            get_action(d["turns"][index+1]["dialogue_acts"]))
                    else:
                        data["target_text"].append(
                            d["turns"][index+1]["utterance"])

    json.dump(data, open("convlab/policy/ussT5/emowoz-test.json", 'w'), indent=2)
    return data


def get_action(dialogue_acts):
    acts = []
    for _, act in dialogue_acts.items():
        for a in act:
            acts.append(
                f"{a['domain'].capitalize()}-{a['intent'].capitalize()}")
    if not acts:
        return "None"
    return ','.join(acts)


def generate_result(model_checkpoint, data, stop=-1):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    results = []
    i = 0
    print("stop", stop)
    for input_text, target_text in tqdm(zip(data["input_text"], data["target_text"]), ascii=True):
        if stop > 0 and i > stop:
            break
        i += 1
        inputs = tokenizer([input_text], return_tensors="pt", padding=True)
        output = model.generate(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                do_sample=False)
        output = tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]

        if "satisfaction score" in input_text:
            output = tri_convert(output)
        results.append({"input_text": input_text,
                        "preds": output,
                        "label": target_text})
    json.dump(results, open(os.path.join(
        model_checkpoint, "emowoz_result.json"), 'w'), indent=2)
    return results


def read_result(result):
    d = {}
    for d_name in ["satisfaction score", "utterance generation", "action prediction"]:
        d[d_name] = {"preds": [], "label": []}
    for r in result:
        for d_name in ["satisfaction score", "utterance generation", "action prediction"]:
            if d_name in r["input_text"]:
                d[d_name]["preds"].append(r["preds"])
                d[d_name]["label"].append(r["label"])
    return d


def satisfaction(model, d):
    # satisfaction
    all_sentiment = ["Neutral", "Negative", "Positive"]
    print(all_sentiment)
    tri_f1 = metrics.f1_score(
        d["satisfaction score"]["label"],
        d["satisfaction score"]["preds"], average="macro")
    sep_f1 = metrics.f1_score(
        d["satisfaction score"]["label"],
        d["satisfaction score"]["preds"], average=None, labels=all_sentiment)
    cm = metrics.confusion_matrix(
        d["satisfaction score"]["label"],
        d["satisfaction score"]["preds"], normalize="true", labels=all_sentiment)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=all_sentiment)
    disp.plot()
    r = {"tri_f1": float(tri_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    print(r)
    time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
    plt.savefig(os.path.join(model, f"{time}-emowoz.png"))


def utterance(model, d):
    bleu_metric = load_metric("sacrebleu")
    labels = [[utt] for utt in d["utterance generation"]["label"]]

    bleu_score = bleu_metric.compute(
        predictions=d["utterance generation"]["preds"],
        references=labels,
        force=True)
    print(f"{model} bleu_score", bleu_score)


def action(model, d):
    score = {}
    for preds, label in zip(d["action prediction"]["preds"], d["action prediction"]["label"]):
        s = f1_score(preds, label)
        for n, v in s.items():
            if n not in score:
                score[n] = []
            score[n].append(v)
    print(f"{model} action")
    for n, v in score.items():
        print(n, np.mean(v))


def f1_score(prediction, label):
    score = {}
    tp = 0
    pre = prediction.split(',')
    lab = label.split(',')
    for p in pre:
        if p in lab:
            tp += 1
    score["precision"] = tp/len(pre)
    score["recall"] = tp/len(lab)
    score["F1"] = 0
    if score["precision"]+score["recall"] > 0:
        score["F1"] = 2*score["precision"]*score["recall"] / \
            (score["precision"]+score["recall"])
    if pre == lab:
        score["acc"] = 1
    else:
        score["acc"] = 0
    return score


def main():
    args = arg_parser()
    if args.gen_file:
        d = read_result(json.load(open(args.gen_file)))
    else:
        data = build_data(load_experiment_dataset(args.data)["test"])
        results = generate_result(args.model, data, args.stop)
        d = read_result(results)
    model = args.model
    satisfaction(model, d)
    utterance(model, d)
    action(model, d)


if __name__ == "__main__":
    main()
