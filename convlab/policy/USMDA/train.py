import os
import random
from argparse import ArgumentParser
import json

import numpy as np
import torch
from datasets import load_metric, Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="",
                        help="input data")
    parser.add_argument("--batch", type=int, default=2,
                        help="batch size")

    return parser.parse_args()


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def read_data(data_dir):
    print("data_dir", data_dir)
    subfix = {"train": "trn", "validation": "dev", "test": "tst"}
    files = {}
    data = {}
    for data_split, sub in subfix.items():
        data[data_split] = parse_data(json.load(
            open(os.path.join(data_dir, f"emotion-detection-{sub}.json"))))

    return data


def parse_data(data):
    emo2label = {
        "Neutral": 0,
        "Scared": 1,
        "Mad": 1,
        "Sad": 1,
        "Joyful": 2,
        "Peaceful": 2,
        "Powerful": 2
    }
    d = []
    for episode in data["episodes"]:
        for scene in episode["scenes"]:
            for r in range(len(scene["utterances"])-1):
                text = ' '.join([scene["utterances"][r]["transcript"],
                                scene["utterances"][r+1]["transcript"]])
                label = emo2label.get(
                    scene["utterances"][r+1]["emotion"], "Neutral")
                d.append({"label": label, "text": text})

    return d


def main():
    args = arg_parser()
    base_name = "convlab/policy/USMDA"
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=3)
    metric = load_metric("accuracy")

    fp16 = False
    if torch.cuda.is_available():
        print("use cuda")
        fp16 = True
        model.to("cuda")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    emory_data = read_data(args.data)
    folder_name = os.path.join(base_name, "data")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    json.dump(emory_data, open(os.path.join(folder_name, "data.json"), 'w'))

    data = {}
    for data_split, d in emory_data.items():
        d = Dataset.from_list(d)
        data[data_split] = d.map(tokenize_function, batched=True)

    model_dir = os.path.join(base_name, "model")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        evaluation_strategy="epoch",
        num_train_epochs=2,
        fp16=fp16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=compute_metrics,)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
