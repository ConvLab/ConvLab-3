import argparse
import logging
import math
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

import copy, operator
from queue import PriorityQueue
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
from convlab2.e2e.soloist.multiwoz.config import global_config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def cuda_(var):
    return var.cuda() if cfg.cuda and torch.cuda.is_available() else var


def tensor(var):
    return cuda_(torch.tensor(var))

class SOLOIST:

    def __init__(self) -> None:
        
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path,config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        print('model loaded!')

        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

    def generate(self, inputs):

        self.model.eval()
        inputs = self.tokenizer([inputs])
        input_ids = tensor(inputs['input_ids'])
        # generated_tokens = self.model.generate(input_ids = input_ids, max_length = cfg.max_length, num_beams = cfg.num_beams)
        generated_tokens = self.model.generate(input_ids = input_ids, max_length = cfg.max_length, top_p=cfg.top_p)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return decoded_preds[0]

    
    def train_loop(self):

        def preprocess_function(examples):
            contextes = examples['Context']
            responses = examples['Response']
            belief = examples['Belief']
            responses_labels = []
            inputs = []

            for context, response, kb in zip(contextes, responses, belief):
                if cfg.no_kb:
                    inputs.append(context + ' => ')
                else:

                    if cfg.format_version == 'e2e':
                        context = ' EOS '.join(context.split(' EOS ')[-10:])
                        _input = context
                    
                    if cfg.format_version == 'e2e+lm':
                        context = ' EOS '.join(context.split(' EOS ')[-10:])
                        inputs.append('[E2E] ' + context)
                        responses_labels.append(response)
                        inputs.append('[LM] ' + context )
                        responses_labels.append(response.split(' EOS ')[1])
                        continue
                    
                    if cfg.format_version == 'v2':
                        _input = kb + context
                    
                    if cfg.format_version == 'v3':
                        _input = ''
                        context = context.split(' EOS ')
                        for idx, turn in enumerate(context):
                            if idx % 2 == 0:
                                _input += 'user : ' + turn.strip()
                            else:
                                _input += ' system : ' + turn.strip()
                        _input = _input + ' <|Knowledge|> ' + kb

                    if cfg.format_version == 'v4':
                        _input = ''
                        context = context.split(' EOS ')
                        for idx, turn in enumerate(context):
                            if idx % 2 == 0:
                                _input += 'user : ' + turn.strip()
                            else:
                                _input += ' system : ' + turn.strip()
                        _input = kb + _input
                    
                    inputs.append(_input)
                    responses_labels.append(response)
            model_inputs = self.tokenizer(inputs, max_length=cfg.max_length, padding="max_length", truncation=True)

            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(responses_labels, max_length=cfg.max_target_length, padding="max_length", truncation=True)

            
            if cfg.ignore_pad_token_for_loss:
                labels["labels"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["labels"]
            return model_inputs

        raw_datasets = load_dataset(cfg.dataset_name)
        column_names = ['Context','Response','Belief']
        lm_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            num_proc=cfg.preprocessing_num_workers,
            load_from_cache_file=False,
            desc=f"Processing dataset",
        )

        train_dataset = lm_datasets["test"]
        # train_dataset = lm_datasets["validation"]
        eval_dataset = lm_datasets["test"]
        test_dataset = lm_datasets["test"]
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.") 

        label_pad_token_id = -100 if cfg.ignore_pad_token_for_loss else self.tokenizer.pad_token_id

        accelerator = Accelerator()
        logger.info(accelerator.state)
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )


        train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=cfg.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=cfg.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=cfg.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

        # Prepare everything with our `accelerator`.
        self.model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, test_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        else:
            cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=cfg.max_train_steps,
        )

        # Metric

        # Train!
        total_batch_size = cfg.per_device_train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {cfg.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        global_steps = 0
        tr_loss, logging_loss = 0.0, 0.0
        for epoch in range(cfg.num_train_epochs):
            self.model.train()
            # for step, batch in enumerate(train_dataloader):
            for step, batch in enumerate(train_dataloader):
                global_steps += 1            
                outputs = self.model(**batch)
                loss = outputs.loss
                loss = loss / cfg.gradient_accumulation_steps
                tr_loss += loss.item()
                accelerator.backward(loss)
                
                if step % cfg.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                if completed_steps >= cfg.max_train_steps:
                    break

                if step % cfg.logging_steps == 0:
                    logger.info(f"  EVALERR:  {(tr_loss - logging_loss)/float(cfg.logging_steps)}")
                    logging_loss = tr_loss
                    progress_bar.update(cfg.logging_steps)

                if cfg.output_dir is not None and global_steps % cfg.save_steps == 0 and global_steps > 0:
                    
                    accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process:               
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(cfg.output_dir, '{}-{}'.format(checkpoint_prefix, global_steps))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

                        self.tokenizer.save_pretrained(output_dir)
                        torch.save(cfg, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)


    