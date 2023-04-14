import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="model name")
    parser.add_argument("--data", type=str)
    parser.add_argument("--gen-file", type=str)
    return parser.parse_args()


def bi_f1(x):
    if x in ['1', '2']:
        return 0
    elif x in ['3', '4', '5']:
        return 1
    else:
        return 0


def tri_convert(x):
    if x == '3':
        return "Neutral"
    if x in ['1', '2']:
        return "Negative"
    if x in ['4', '5']:
        return "Positive"
    return "Neutral"


def bi_check(p, l):
    negative = ['1', '2']
    positive = ['3', '4', '5']
    if p in negative and l in negative:
        return 1
    if p in positive and l in positive:
        return 1

    return 0


def read_result(result):
    preds = {'bi': [], "five": [], 'tri': []}
    label = {'bi': [], "five": [], 'tri': []}
    for r in result:
        p = r["preds"]
        l = r["label"]
        preds["five"].append(p)
        preds["bi"].append(bi_f1(p))
        preds["tri"].append(tri_convert(p))

        label["five"].append(l)
        label["bi"].append(bi_f1(l))
        label["tri"].append(tri_convert(l))
    return preds, label


def generate_result(model_checkpoint, data):
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    data = pd.read_csv(data, index_col=False).astype(str)
    results = []
    for input_text, target_text in tqdm(zip(data["input_text"], data["target_text"]), ascii=True):
        if "satisfaction score" in input_text:
            inputs = tokenizer([input_text], return_tensors="pt", padding=True)
            output = model.generate(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    do_sample=False)
            output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if len(output) > 1:
                print(output)
                output = "illegal"

            results.append({"input_text": input_text,
                            "preds": output,
                            "label": target_text})
    json.dump(results, open(os.path.join(
        model_checkpoint, "uss_result.json"), 'w'))
    return results


def main():
    args = arg_parser()
    if args.gen_file:
        preds, label = read_result(json.load(open(args.gen_file)))
    else:
        results = generate_result(args.model, args.data)
        preds, label = read_result(results)

    macro_f1 = metrics.f1_score(label["five"], preds["five"], average="macro")
    tri_f1 = metrics.f1_score(label["tri"], preds["tri"], average="macro")
    f1 = metrics.f1_score(label["bi"], preds["bi"])
    sep_f1 = metrics.f1_score(
        label["five"], preds["five"], average=None,
        labels=['1', '2', '3', '4', '5'])
    cm = metrics.confusion_matrix(
        label["five"], preds["five"], normalize="true",
        labels=['1', '2', '3', '4', '5'])
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['1', '2', '3', '4', '5'])
    disp.plot()
    r = {"macro_f1": float(macro_f1),
         "tri_f1": float(tri_f1),
         "bi_f1": float(f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    print(r)
    dirname = "convlab/policy/uss-t5/"
    time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
    plt.savefig(os.path.join(args.model, f"{time}-satisfied.png"))


if __name__ == "__main__":
    main()
