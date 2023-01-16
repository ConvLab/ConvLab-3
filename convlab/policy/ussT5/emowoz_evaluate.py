import json
import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from convlab.policy.tus.unify.util import create_goal, load_experiment_dataset
from convlab.policy.ussT5.evaluate import tri_convert


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
    for sentiment, index in json.load(open("convlab/policy/emoTUS/sentiment.json")).items():
        sentiments[int(index)] = sentiment
    data = {"input_text": [], "target_text": []}
    prefix = "satisfaction score: "
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
                data["target_text"].append(
                    sentiments[d["turns"][index+1]["emotion"][-1]["sentiment"]])
    return data


def generate_result(model_checkpoint, data, stop=-1):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    results = []
    i = 0
    print("stop", stop)
    for input_text, target_text in tqdm(zip(data["input_text"], data["target_text"]), ascii=True):
        if stop > 0 and i > stop:
            break
        if "satisfaction score" in input_text:
            i += 1
            inputs = tokenizer([input_text], return_tensors="pt", padding=True)
            output = model.generate(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    do_sample=False)
            output = tokenizer.batch_decode(
                output, skip_special_tokens=True)[0]
            if len(output) > 1:
                print(output)
                output = "illegal"

            results.append({"input_text": input_text,
                            "preds": tri_convert(output),
                            "label": target_text})
    json.dump(results, open(os.path.join(
        model_checkpoint, "emowoz_result.json"), 'w'))
    return results


def read_result(result):
    preds = []
    label = []
    for r in result:
        preds.append(r["preds"])
        label.append(r["label"])
    return preds, label


def main():
    args = arg_parser()
    if args.gen_file:
        preds, label = read_result(json.load(open(args.gen_file)))
    else:
        data = build_data(load_experiment_dataset(args.data)["test"])
        results = generate_result(args.model, data, args.stop)
        preds, label = read_result(results)
    all_sentiment = ["Neutral", "Negative", "Positive"]
    print(all_sentiment)
    tri_f1 = metrics.f1_score(label, preds, average="macro")
    sep_f1 = metrics.f1_score(label, preds, average=None, labels=all_sentiment)
    cm = metrics.confusion_matrix(
        label, preds, normalize="true", labels=all_sentiment)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=all_sentiment)
    disp.plot()
    r = {"tri_f1": float(tri_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    print(r)
    time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
    plt.savefig(os.path.join(args.model, f"{time}-emowoz.png"))


if __name__ == "__main__":
    main()
