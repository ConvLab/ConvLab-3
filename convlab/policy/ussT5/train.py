import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from sklearn.model_selection import train_test_split
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer)

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


class ForT5Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index]).squeeze()
        target_ids = torch.tensor(self.targets[index]).squeeze()

        return {"input_ids": input_ids, "labels": target_ids}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="act-sat-utt",
                        help="act-sat, act-sat-utt, act-sat_no-alt, or act-sat-utt_no-alt")
    parser.add_argument("--data", type=str, default="",
                        help="input data")
    parser.add_argument("--batch", type=int, default=8,
                        help="batch size")

    return parser.parse_args()


def main():
    set_seed(0)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def preprocess_function(examples):
        inputs = examples["input_text"].to_list()
        targets = examples["target_text"].to_list()
        model_inputs = tokenizer(inputs, text_target=targets,
                                 max_length=512, truncation=True)
        return model_inputs

    args = arg_parser()
    base_name = "convlab/policy/ussT5"
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metric = load_metric("sacrebleu")

    output_dir = os.path.join(base_name, "experiments", args.task)
    # f"{datetime.now().strftime('%y-%m-%d-%H-%M')}")

    raw_data = pd.read_csv(args.data, index_col=False).astype(str)
    data = {"train": None, "validation": None, "test": None}
    train_set, data["test"] = train_test_split(raw_data, test_size=0.1)
    data["train"], data["validation"] = train_test_split(
        train_set, test_size=0.1)
    folder_name = os.path.join(base_name, "data", args.task)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print("Building data...")
    for data_type in data:
        data[data_type].to_csv(os.path.join(folder_name, f"{data_type}.csv"))
        data[data_type] = preprocess_function(data[data_type])
        data[data_type] = ForT5Dataset(inputs=data[data_type]["input_ids"],
                                       targets=data[data_type]["labels"])

    fp16 = False
    if torch.cuda.is_available():
        print("use cuda")
        fp16 = True
        model.to("cuda")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
