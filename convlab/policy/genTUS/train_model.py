import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
import transformers
from datasets import Dataset, load_metric
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration, BartTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

os.environ["WANDB_DISABLED"] = "true"

METRIC = load_metric("sacrebleu")
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base")
TOKENIZER.add_tokens(["<?>"])
MAX_IN_LEN = 500
MAX_OUT_LEN = 500


def arg_parser():
    parser = ArgumentParser()
    # data_name, dial_ids_order, split2ratio
    parser.add_argument("--model-type", type=str, default="unify",
                        help="unify or multiwoz")
    parser.add_argument("--data-name", type=str, default="multiwoz21",
                        help="multiwoz21, sgd, tm1, tm2, tm3, sgd+tm, or all")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-checkpoint", type=str,
                        default="facebook/bart-base")
    return parser.parse_args()


def gentus_compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = TOKENIZER.batch_decode(
        preds, skip_special_tokens=True, max_length=MAX_OUT_LEN)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(
        labels, skip_special_tokens=True, max_length=MAX_OUT_LEN)

    act, text = postprocess_text(decoded_preds, decoded_labels)

    result = METRIC.compute(
        # predictions=decoded_preds, references=decoded_labels)
        predictions=text["preds"], references=text["labels"])
    result = {"bleu": result["score"]}
    f1_scores = f1_measure(pred_acts=act["preds"], label_acts=act["labels"])
    for s in f1_scores:
        result[s] = f1_scores[s]

    result = {k: round(v, 4) for k, v in result.items()}
    return result


def postprocess_text(preds, labels):
    act = {"preds": [], "labels": []}
    text = {"preds": [], "labels": []}

    for pred, label in zip(preds, labels):
        model_output = parse_output(pred.strip())
        label_output = parse_output(label.strip())
        if len(label_output["text"]) < 1:
            continue
        act["preds"].append(model_output.get("action", []))
        text["preds"].append(model_output.get("text", pred.strip()))
        act["labels"].append(label_output["action"])
        text["labels"].append([label_output["text"]])

    return act, text


def parse_output(in_str):
    in_str = in_str.replace('<s>', '').replace('<\\s>', '')
    try:
        output = json.loads(in_str)
    except:
        # print(f"invalid action {in_str}")
        output = {"action": [], "text": ""}
    return output


def f1_measure(pred_acts, label_acts):
    result = {"precision": [], "recall": [], "f1": []}
    for pred, label in zip(pred_acts, label_acts):
        r = tp_fn_fp(pred, label)
        for m in result:
            result[m].append(r[m])
    for m in result:
        result[m] = sum(result[m])/len(result[m])

    return result


def tp_fn_fp(pred, label):
    tp, fn, fp = 0.0, 0.0, 0.0
    precision, recall, f1 = 0, 0, 0
    for p in pred:
        if p in label:
            tp += 1
        else:
            fp += 1
    for l in label:
        if l not in pred:
            fn += 1
    if (tp+fp) > 0:
        precision = tp / (tp+fp)
    if (tp+fn) > 0:
        recall = tp/(tp+fn)
    if (precision + recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)

    return {"precision": precision, "recall": recall, "f1": f1}


class TrainerHelper:
    def __init__(self, tokenizer, max_input_length=500, max_target_length=500):
        print("transformers version is: ", transformers.__version__)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.base_name = "convlab/policy/genTUS"
        self.dir_name = ""

    def _get_data_folder(self, model_type, data_name, dial_ids_order=0, split2ratio=1):
        # base_name = "convlab/policy/genTUS/unify/data"
        if model_type not in ["unify", "multiwoz"]:
            print("Unknown model type. Currently only support unify and multiwoz")
        self.dir_name = f"{data_name}_{dial_ids_order}_{split2ratio}"
        return os.path.join(self.base_name, model_type, 'data', self.dir_name)

    def get_model_folder(self, model_type):
        folder_name = os.path.join(
            self.base_name, model_type, "experiments", self.dir_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name

    def parse_data(self, model_type, data_name, dial_ids_order=0, split2ratio=1):
        data_folder = self._get_data_folder(
            model_type, data_name, dial_ids_order, split2ratio)

        raw_data = {}
        for d_type in ["train", "validation", "test"]:
            f_name = os.path.join(data_folder, f"{d_type}.json")
            raw_data[d_type] = json.load(open(f_name))

        tokenized_datasets = {}
        for data_type, data in raw_data.items():
            tokenized_datasets[data_type] = Dataset.from_dict(
                self._preprocess(data["dialog"]))

        return tokenized_datasets

    def _preprocess(self, examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        if isinstance(examples, dict):
            examples = [examples]
        for example in tqdm(examples):
            inputs = self.tokenizer(example["in"],
                                    max_length=self.max_input_length,
                                    truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(example["out"],
                                        max_length=self.max_target_length,
                                        truncation=True)
            for key in ["input_ids", "attention_mask"]:
                model_inputs[key].append(inputs[key])
            model_inputs["labels"].append(labels["input_ids"])

        return model_inputs


def train(model_type, data_name, dial_ids_order, split2ratio, batch_size=16, max_input_length=500, max_target_length=500, model_checkpoint="facebook/bart-base"):
    tokenizer = TOKENIZER

    train_helper = TrainerHelper(
        tokenizer=tokenizer, max_input_length=max_input_length, max_target_length=max_target_length)
    data = train_helper.parse_data(model_type=model_type,
                                   data_name=data_name,
                                   dial_ids_order=dial_ids_order,
                                   split2ratio=split2ratio)

    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    fp16 = False
    if torch.cuda.is_available():
        fp16 = True

    model_dir = os.path.join(
        train_helper.get_model_folder(model_type),
        f"{datetime.now().strftime('%y-%m-%d-%H-%M')}")

    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=fp16,
        push_to_hub=False,
        generation_max_length=max_target_length,
        logging_dir=os.path.join(model_dir, 'log')
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True)

    # customize this trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=gentus_compute_metrics)
    print("start training...")
    trainer.train()
    print("saving model...")
    trainer.save_model()


def main():
    args = arg_parser()
    print("---> data_name", args.data_name)
    train(model_type=args.model_type,
          data_name=args.data_name,
          dial_ids_order=args.dial_ids_order,
          split2ratio=args.split2ratio,
          batch_size=args.batch_size,
          max_input_length=MAX_IN_LEN,
          max_target_length=MAX_OUT_LEN,
          model_checkpoint=args.model_checkpoint)


if __name__ == "__main__":
    main()
    # sgd+tm: 46000
