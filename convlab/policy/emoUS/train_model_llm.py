import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import transformers
from datasets import Dataset, load_metric
from tqdm import tqdm
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader

from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# os.environ["WANDB_DISABLED"] = "true"


def arg_parser():
    parser = ArgumentParser()
    # data_name, dial_ids_order, split2ratio
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model-type", type=str, default="unify",
                        help="unify or multiwoz")
    parser.add_argument("--data-name", type=str, default="emowoz",
                        help="emowoz or dialmage")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-checkpoint", type=str,
                        default="facebook/bart-base")
    parser.add_argument("--fine-tune", action="store_true")
    return parser.parse_args()


METRIC = load_metric("sacrebleu")
# TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base")
# TOKENIZER.add_tokens(["<?>"])
MAX_IN_LEN = 500
MAX_OUT_LEN = 500


def get_model(model_path):
    # Initialise models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.add_tokens(["<?>"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        print("use cuda")
        device = "cuda"
        model.to("cuda")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, device, peft_config


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
        self.base_name = "convlab/policy/emoUS"
        self.dir_name = ""

    def _get_data_folder(self, model_type, data_name, dial_ids_order=0, split2ratio=1):
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

    def remove_dialmage_action(self):
        self.dir_name = "fine_tune"
        folder = "convlab/policy/emoUS/unify/data"
        data_name = {
            "emowoz": "EmoUS_emowoz_0_1",
            "dialmage": "EmoUS_dialmage_0_1_emotion_only"}
        data = {}
        for d, d_n in data_name.items():
            data[d] = {}
            for d_type in ["train", "validation", "test"]:
                f_name = os.path.join(folder, d_n, f"{d_type}.json")
                data[d][d_type] = json.load(open(f_name))

        tokenized_datasets = {}
        for d_n, d in data.items():
            tokenized_datasets[d_n] = {}
            for s_d_n, s_d in d.items():
                tokenized_datasets[d_n][s_d_n] = Dataset.from_dict(
                    self._preprocess(s_d["dialog"]))
        return tokenized_datasets

    def _preprocess(self, examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        if isinstance(examples, dict):
            examples = [examples]
        for example in tqdm(examples):
            inputs = self.tokenizer(example["in"],
                                    max_length=self.max_input_length,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt")

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(example["out"],
                                        max_length=self.max_target_length,
                                        truncation=True,
                                        return_tensors="pt",
                                        padding="max_length")
            for key in ["input_ids", "attention_mask"]:
                model_inputs[key].append(inputs[key])
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"].append(labels["input_ids"])

        return model_inputs


def train(model_type,
          data_name,
          dial_ids_order,
          split2ratio,
          batch_size=16,
          max_input_length=500,
          max_target_length=500,
          model_checkpoint="facebook/bart-base",
          learning_rate=2e-5,
          num_epochs=5):
    model, tokenizer, device, peft_config = get_model(model_checkpoint)
    # tokenizer = TOKENIZER

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, max_length=MAX_OUT_LEN)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, max_length=MAX_OUT_LEN)

        act, text = postprocess_text(decoded_preds, decoded_labels)

        result = METRIC.compute(
            # predictions=decoded_preds, references=decoded_labels)
            predictions=text["preds"], references=text["labels"])
        result = {"bleu": result["score"]}
        f1_scores = f1_measure(
            pred_acts=act["preds"], label_acts=act["labels"])
        for s in f1_scores:
            result[s] = f1_scores[s]

        result = {k: round(v, 4) for k, v in result.items()}
        return result

    train_helper = TrainerHelper(
        tokenizer=tokenizer, max_input_length=max_input_length, max_target_length=max_target_length)
    data = train_helper.parse_data(model_type=model_type,
                                   data_name=data_name,
                                   dial_ids_order=dial_ids_order,
                                   split2ratio=split2ratio)
    data_loader = {}
    for x in data:
        data_loader[x] = DataLoader(
            data[x], shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(data_loader["train"]) * num_epochs),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(data_loader["train"])):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        # eval_preds = []
        for step, batch in enumerate(tqdm(data_loader["validation"])):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            # eval_preds.extend(
            #     tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

        eval_epoch_loss = eval_loss / len(data_loader["validation"])
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(data_loader["train"])
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch}: {train_ppl} {train_epoch_loss} {eval_ppl} {eval_epoch_loss}")

    model_dir = os.path.join(
        train_helper.get_model_folder(model_type),
        f"{datetime.now().strftime('%y-%m-%d-%H-%M')}")

    peft_model_id = f"{model_dir}_{peft_config.peft_type}_{peft_config.task_type}"
    model.save_pretrained(peft_model_id)


def main():
    args = arg_parser()
    print("---> data_name", args.data_name)
    train(
        model_type=args.model_type,
        data_name=args.data_name,
        dial_ids_order=args.dial_ids_order,
        split2ratio=args.split2ratio,
        batch_size=args.batch_size,
        max_input_length=MAX_IN_LEN,
        max_target_length=MAX_OUT_LEN,
        model_checkpoint=args.model_checkpoint
    )


if __name__ == "__main__":
    main()
    # sgd+tm: 46000
