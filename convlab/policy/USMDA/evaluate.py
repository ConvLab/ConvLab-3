import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="model name")
    parser.add_argument("--data", type=str)
    parser.add_argument("--gen-file", type=str)
    return parser.parse_args()


def generate_result(model_checkpoint, data):
    result = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint)
    data = pd.read_csv(data, index_col=False).astype(str)
    # Neutral: 0, Negative: 1, Positive: 2
    t2i = {'3': 0, '1': 1, '2': 1, '4': 2, '5': 2}
    prefix = "satisfaction score: "
    for input_text, target_text in tqdm(zip(data["input_text"], data["target_text"]), ascii=True):
        if prefix in input_text:
            text = input_text.replace(prefix, '')
            target = t2i[target_text]
            model_input = tokenizer(
                [text], return_tensors="pt", padding=True)
            output = model(input_ids=model_input["input_ids"],
                           attention_mask=model_input["attention_mask"])
            output = int(np.argmax(output, axis=-1))
            result.append({"input_text": text,
                           "preds": output,
                           "label": target})
    json.dump(result, open(os.path.join(
        model_checkpoint, "uss_result.json"), 'w'))
    return result


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
        results = generate_result(args.model, args.data)
        preds, label = read_result(results)

    macro_f1 = metrics.f1_score(label, preds, average="macro")
    sep_f1 = metrics.f1_score(
        label, preds, average=None,
        labels=[0, 1, 2])
    cm = metrics.confusion_matrix(
        label, preds, normalize="true",
        labels=[0, 1, 2])
    print("Neutral: 0, Negative: 1, Positive: 2")
    print("cm", cm)
    print("f1", sep_f1)
    print("macro", macro_f1)


if __name__ == "__main__":
    main()
