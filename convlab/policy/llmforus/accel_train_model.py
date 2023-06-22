import argparse
import json
import math
import os
import pickle
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from datasets import Dataset
from peft import (LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_config,
                  get_peft_model)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          SchedulerType, get_scheduler)


def arg_parser():
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path", type=str, default="./hf_models", help="Path to the model file",)
    parser.add_argument(
        "--train_data_path", type=str, default="./hf_models", help="Path to the train data file",)
    parser.add_argument(
        "--val_data_path", type=str, default="./hf_models", help="Path to the val data file",)
    parser.add_argument(
        "--resume", type=str, default="", help="Path to the saved checkpoint",)
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.", )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Batch size (per device) for the evaluation dataloader.",)
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"], )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--logfile", type=str, default='./log.txt', help="Path to the log file", )
    parser.add_argument(
        "--outputdir", type=str, default='./exp/clip_vlm', help="Path to the output dir",)
    parser.add_argument(
        "--log_interval", type=int, default=100, help="log interval", )
    parser.add_argument(
        "--topn", type=int, default=1, help="Top n from the list to use",)
    parser.add_argument(
        "--ontology", type=str, default="", help="KB for biasing",)
    parser.add_argument(
        "--maxKBsize", type=int, default=10, help="Size of the biasing list to use",)
    parser.add_argument(
        "--KBdrop", type=float, default=0.5, help="Drop ratio for true biasing entities",)
    parser.add_argument(
        "--tag", type=str, default="", help="Schema config", )
    args = parser.parse_args()
    return args


def set_seed(seed=1):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = arg_parser()
    # maybe we should put seed in args
    set_seed()
    train(args)


def logging(s, logfile="", logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def get_model(model_path, resume=""):
    # Initialise models
    # print(f"Get model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16)

    if model_path != "gpt2":
        # Use LoRA
        if resume != "":
            # print("Resuming from {}".format(resume))
            model = PeftModel.from_pretrained(
                model, resume, is_trainable=True)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model


def get_optimizer(model: AutoModelForCausalLM, weight_decay=0.0, learning_rate=5e-5):
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


def get_lr_scheduler(optimizer, trainsize, gradient_accumulation_steps=1, batch_size=2, num_train_epochs=3, lr_scheduler_type="linear", num_warmup_steps=0):
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        trainsize / gradient_accumulation_steps / batch_size)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    return lr_scheduler


class LLamaDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, index):
        return {"input_ids": np.array(self.input_ids[index]),
                "attention_mask": np.array(self.attention_mask[index]),
                "labels": np.array(self.labels[index])}

    def __len__(self):
        return len(self.input_ids)


def get_data(data_path, input_tokenizer, output_tokenizer):
    # TODO -> setup for emoUS
    # Initialise dataloaders
    data = {"in": [], "out": []}

    with open(data_path) as fin:
        text = json.load(fin)
        for dialog in text["dialog"]:
            inputs = input_tokenizer(
                dialog["in"], max_length=400, truncation=True, return_tensors="pt")
            labels = output_tokenizer(
                dialog["out"], max_length=100, truncation=True, return_tensors="pt")
            labels = labels["input_ids"][0]
            data["in"].append(
                torch.cat([inputs["input_ids"][0], labels]))
            data["out"].append(
                torch.cat([inputs["input_ids"][0]*0-1, labels]))
    data["in"] = pad_sequence(
        data["in"], batch_first=True, padding_value=0)
    data["out"] = pad_sequence(
        data["out"], batch_first=True, padding_value=-1)
    print("-----> shape", data["in"].shape, data["out"].shape)
    attnmask = data["in"] != 0
    inputs = {"input_ids": data["in"][:, :-1],
              "attention_mask": attnmask[:, :-1]}

    # change to list like?
    # dataset = Dataset.from_dict({"input_ids": inputs["input_ids"].tolist(),
    #                              "attention_mask": inputs["attention_mask"].tolist(),
    #                              "labels": data["out"].tolist()}).with_format("torch")
    dataset = LLamaDataset(input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           labels=data["out"])
    return dataset


def train(args):
    print("gradient_accumulation_steps", args.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    output_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    output_tokenizer.add_eos_token = True
    output_tokenizer.add_bos_token = False

    llama_data = {"train": get_data(args.train_data_path, tokenizer, output_tokenizer),
                  "validation": get_data(args.val_data_path, tokenizer, output_tokenizer)}

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_training_loop(batch_size, data):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        model = get_model(args.model_path, args.resume)
        model.to(accelerator.device)

        accelerator.print("------> batch_size:", batch_size)
        accelerator.print(f"Preparing training datasets...")

        trainloader = DataLoader(
            data["train"],
            batch_size=batch_size,
            shuffle=True)
        accelerator.print(f"Preparing validation datasets...")
        validloader = DataLoader(
            data["validation"],
            batch_size=batch_size)

        trainsize = len(trainloader)

        # Initialise criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        accelerator.print(f"Preparing optimizer...")
        optimizer = get_optimizer(model, args.weight_decay, args.learning_rate)
        accelerator.print(f"Preparing scheduler...")
        lr_scheduler = get_lr_scheduler(optimizer, trainsize, args.gradient_accumulation_steps,
                                        batch_size, args.num_train_epochs, args.lr_scheduler_type, args.num_warmup_steps)
        accelerator.print(
            f"Preparing model, optimizer, and scheduler for accelerator...")
        model, optimizer, trainloader, validloader, lr_scheduler = accelerator.prepare(
            model, optimizer, trainloader, validloader, lr_scheduler)
        accelerator.print(f"Test saving...")
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.outputdir)
        model = accelerator.prepare(model)
        bestvalloss = 100000
        accelerator.print(f"Starting training...")
        for epoch in range(args.num_train_epochs):
            accelerator.print(f"Epoch {epoch}...")
            model.train()
            optimizer.zero_grad()
            start = time.time()
            for i, batch in enumerate(trainloader):
                with accelerator.accumulate(model):
                    inputs = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"]}
                    labels = batch["labels"]

                    output = model(**inputs, return_dict=True)
                    logits = output.logits
                    loss = criterion(logits.view(-1, logits.size(-1)),
                                     labels[:, 1:].reshape(-1))
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Evaluation starts
            model.eval()
            with torch.no_grad():
                total_tokens = 0
                total_loss = 0.
                for i, batch in enumerate(validloader):
                    inputs = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"]}
                    labels = batch["labels"]
                    output = model(**inputs, return_dict=True)
                    logits = output.logits
                    loss = criterion(logits.view(-1, logits.size(-1)),
                                     labels[:, 1:].reshape(-1))
                    tokens = (labels != -1).sum()
                    total_tokens += tokens
                    total_loss += loss.item() * tokens
                val_loss = total_loss / total_tokens
                val_ppl = math.exp(val_loss)
                logging(f"Epoch {epoch} | Validation PPL: {val_ppl}",
                        logfile=args.logfile)
            # if val_loss < bestvalloss:
                # torch.save(model.state_dict(), os.path.join(args.outputdir, f"snapshot.ep.{epoch}"))
                # if args.model_path != "gpt2":
            accelerator.wait_for_everyone()
            logging(f"Saving best model at Epoch {epoch}",
                    logfile=args.logfile)
            # model.save_pretrained(args.outputdir)
            model = accelerator.unwrap_model(model)
            model.save_pretrained(args.outputdir)
            model = accelerator.prepare(model)
            # else:
            #     torch.save(model.state_dict(), os.path.join(
            #         args.outputdir, "checkpoint.best".format(epoch)))
            #     torch.save(model.state_dict(), os.path.join(
            #         args.outputdir, "checkpoint.ep.{}".format(epoch)))
            # bestvalloss = val_loss
            logging("Current learning rate {}".format(
                optimizer.param_groups[0]["lr"]),
                logfile=args.logfile)
            elasped_time = time.time() - start
            logging(f"Epoch {epoch} took {elasped_time} seconds")
    inner_training_loop(llama_data)


if __name__ == "__main__":
    main()
