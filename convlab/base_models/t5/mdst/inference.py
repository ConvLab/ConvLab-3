#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DST inference turn by turn
Modified from https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from itertools import zip_longest
from functools import reduce

import datasets
import numpy as np
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from convlab.base_models.t5.trainer import ConvLabSeq2SeqTrainer, ConvLabSeq2SeqTrainingArguments
from convlab.base_models.t5.mdst.data_processor import DataProcessor
from convlab.util import load_ontology


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.16.1")

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    truncation_side: Optional[str] = field(
        default="right",
        metadata={"help": "Which side to truncate, left or right."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    model_type: Optional[int] = field(
        default=0,
        metadata={"help": "DST model type. see `data_processor.py`"},
    )
    context_window_size: Optional[int] = field(
        default=100,
        metadata={"help": "Context window size. see `data_processor.py`"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the task, e.g., rg (for rgresponse generation)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the source texts."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the target texts."},
    )
    src_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory containing full_state.json"
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics on (a jsonlines or csv file)."
        },
    )
    metric_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional metric name or file to evaluate the model."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConvLabSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        truncation_side=model_args.truncation_side,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        # use_cache=True is incompatible with gradient checkpointing.
        config.use_cache = False

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    
    logger.info(f'source prefix: "{prefix}"')

    if data_args.source_column is None or data_args.target_column is None:
        raise ValueError(
                f"--source_column' value '{data_args.source_column}' could not be None"+
                f"--target_column' value '{data_args.target_column}' could not be None"
            )

    source_column, target_column = data_args.source_column, data_args.target_column

    max_target_length = data_args.val_max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.generation_max_length is None:
        training_args.generation_max_length = data_args.val_max_target_length

    def preprocess_function(examples):

        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] is not None and examples[target_column][i] is not None:
                inputs.append(examples[source_column][i])
                targets.append(examples[target_column][i])

        inputs = [prefix + inp for inp in inputs]
        if padding:
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        else:
            # truncate each part separated by \n\n respectively
            split_inputs = [inp.split('\n\n') for inp in inputs]
            split_model_inputs = [tokenizer(x, max_length=data_args.max_source_length, padding=False, truncation=True) for x in split_inputs]
            model_inputs = {k: [reduce(lambda x, y: x[:-1]+y, item[k]) for item in split_model_inputs] for k in split_model_inputs[0]}

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # load dialogs and prepare predict_dataset turn by turn
    test_dialogs = json.load(open(data_args.test_file))
    max_turns = max([len(dial['turns']) for dial in test_dialogs])
    ontology = load_ontology(data_args.dataset_name)
    full_state = json.load(open(os.path.join(data_args.src_data_dir, 'full_state.json')))
    data_processor = DataProcessor(model_args.model_type, None, model_args.context_window_size, ontology, full_state)
    for turn_idx in range(0,max_turns,2):
        # turn by turn inference
        samples = data_processor.read_turns_from_dials(test_dialogs, turn_idx)
        predict_dataset = Dataset.from_list(samples)
        column_names = ['domains', 'input', 'output']
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
             predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        
        # Initialize our Trainer
        trainer = ConvLabSeq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Predict
        file_prefix = os.path.splitext(os.path.basename(data_args.test_file))[0]
        logger.info(f"*** Predict {file_prefix} turn {turn_idx}***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix=file_prefix)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics[f"{file_prefix}_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics(file_prefix, metrics)
        trainer.save_metrics(file_prefix, metrics)

        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        data_processor.write_turns_to_dials(test_dialogs, turn_idx, predictions)
        
    if trainer.is_world_process_zero():
        output_prediction_file = os.path.join(training_args.output_dir, f"{file_prefix}_generated_predictions.json")
        with open(output_prediction_file, "w", encoding='utf-8') as writer:
            json.dump(test_dialogs, writer, indent=2)


if __name__ == "__main__":
    main()
