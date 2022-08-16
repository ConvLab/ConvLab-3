import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration, BartTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

os.environ["WANDB_DISABLED"] = "true"


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
    return parser.parse_args()

class Trainer:
    def __init__(self, max_input_length=500, max_target_length=100):
        print("transformers version is: ", transformers.__version__)
        self.metric = load_metric("sacrebleu")
        self.model_checkpoint = "facebook/bart-base"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_checkpoint)
        special_tokens = ["<?>"]
        self.tokenizer.add_tokens(special_tokens)
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

    def _get_model_folder(self, model_type):
        folder_name = os.path.join(self.base_name, model_type, "experiments", self.dir_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name
    
    def parse_data(self, model_type, data_name, dial_ids_order=0, split2ratio=1):
        data_folder = self._get_data_folder(model_type, data_name, dial_ids_order, split2ratio)

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

    def train(self, model_type, datasets, batch_size=16):
        model = BartForConditionalGeneration.from_pretrained(self.model_checkpoint)
        model.resize_token_embeddings(len(self.tokenizer))
        fp16 = False
        if torch.cuda.is_available():
            fp16 = True

        model_dir = os.path.join(
            self._get_model_folder(model_type),
            f"{datetime.now().strftime('%y-%m-%d-%H-%M')}")

        args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=5,
            predict_with_generate=True,
            fp16=fp16,
            push_to_hub=False,
            generation_max_length=400,
            logging_dir=os.path.join(model_dir, 'log')
        )
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=model, padding=True)

        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        print("start training...")
        trainer.train()
        print("saving model...")
        trainer.save_model()

    def _postprocess_text(self, preds, labels):
        act = {"preds": [], "labels": []}
        text = {"preds": [], "labels": []}

        for pred in preds:
            output = self._parse_output(pred.strip())
            act["preds"].append(output["action"])
            text["preds"].append(output["text"])

        for label in labels:
            output = self._parse_output(label.strip())
            act["labels"].append(output["action"])
            text["labels"].append([output["text"]])

        return act, text

    @staticmethod
    def _parse_output(in_str):
        in_str = in_str.replace('<s>', '').replace('<\\s>', '')
        try:
            output = json.loads(in_str)
        except:
            # print(f"invalid action {in_str}")
            output = {"action": [], "text": ""}
        return output


    def _f1_measure(self, pred_acts, label_acts):
        result = {"precision": [], "recall": [], "f1": []}
        for pred, label in zip(pred_acts, label_acts):
            r = self._tp_fn_fp(pred, label)
            for m in result:
                result[m].append(r[m])
        for m in result:
            result[m] = sum(result[m])/len(result[m])

        return result

    @staticmethod
    def _tp_fn_fp(pred, label):
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


    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True, max_length=400)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, max_length=400)

        # Some simple post-processing
        act, text = self._postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(
            predictions=text["preds"], references=text["labels"])
        result = {"bleu": result["score"]}
        f1_scores = self._f1_measure(pred_acts=act["preds"], label_acts=act["labels"])
        for s in f1_scores:
            result[s] = f1_scores[s]

        result = {k: round(v, 4) for k, v in result.items()}
        return result

def main():
    args = arg_parser()
    trainer = Trainer(max_input_length=500, max_target_length=100)
    data = trainer.parse_data(model_type=args.model_type,
                              data_name=args.data_name,
                              dial_ids_order=args.dial_ids_order,
                              split2ratio=args.split2ratio)
    trainer.train(model_type=args.model_type,
                  datasets=data,
                  batch_size=args.batch_size)

if __name__ == "__main__":
    main()
