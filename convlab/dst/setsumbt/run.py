# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
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
"""Run SetSUMBT belief tracker training and evaluation."""

import os
from copy import deepcopy
from shutil import copy2 as copy

import torch
import transformers
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import BertConfig, RobertaConfig

from convlab.dst.setsumbt.datasets import (
    change_batch_size,
    dataloader_sample_dialogues,
    get_dataloader,
    get_distillation_dataloader,
)
from convlab.dst.setsumbt.modeling import SetSUMBTModels, SetSUMBTTrainer
from convlab.dst.setsumbt.utils import (
    get_args,
    get_logger,
    get_model_path,
    setup_ensemble,
    update_args,
)

MODELS = {
    "bert": (BertConfig, "BertTokenizer"),
    "roberta": (RobertaConfig, "RobertaTokenizer"),
    "meta": (RobertaConfig, "RobertaTokenizer"),
}


def main():
    args, config = get_args(MODELS)

    if args.model_type in SetSUMBTModels:
        SetSumbtModel, OntologyEncoderModel, ConfigClass, Tokenizer = SetSUMBTModels[
            args.model_type
        ]
        if args.ensemble:
            SetSumbtModel, _, _, _ = SetSUMBTModels["ensemble"]
    else:
        raise NameError("NotImplemented")

    # Set up output directory
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.mkdir(os.path.join(OUTPUT_DIR, "dataloaders"))
    args.output_dir = OUTPUT_DIR

    # Set pretrained model path to the trained checkpoint
    args, _config = get_model_path(args, ConfigClass)
    config = _config if _config is not None else config

    args = update_args(args, config)

    # Create TensorboardX writer
    tb_writer = SummaryWriter(logdir=args.tensorboard_path)

    # Create logger
    if args.send_logging_emails:
        send_emails = True
        email_subject = "/".join(args.output_dir.split("setsumbt")[1:])
        email_subject = "-".join(
            ["SetSUMBT"] + [i.title() for i in email_subject.split("/") if i]
        )
        email_subject += " Log"
    else:
        send_emails = False
        email_subject = ""
    logger = get_logger(
        args.logging_path, send_emails=send_emails, email_subject=email_subject
    )

    # Get device
    if torch.cuda.is_available() and args.n_gpu > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0

    if args.n_gpu == 0:
        args.fp16 = False

    # Initialise Model
    transformers.utils.logging.set_verbosity_info()
    if not args.do_ensemble_setup:
        model = SetSumbtModel.from_pretrained(args.model_name_or_path, config=config)
        model = model.to(device)

        if args.ensemble:
            args.model_name_or_path = model._get_checkpoint_path(
                args.model_name_or_path, 0
            )

    # Create Tokenizer and embedding model for Data Loaders and ontology
    tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    encoder = OntologyEncoderModel.from_pretrained(
        config.candidate_embedding_model_name, args=args, tokenizer=tokenizer
    )

    transformers.utils.logging.set_verbosity_error()
    if args.do_ensemble_setup:
        # Build all dataloaders
        train_dataloader = get_dataloader(
            args.dataset,
            "train",
            args.train_batch_size,
            tokenizer,
            encoder,
            args.max_dialogue_len,
            args.max_turn_len,
            train_ratio=args.dataset_train_ratio,
            seed=args.seed,
        )
        torch.save(
            train_dataloader,
            os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader"),
        )
        dev_dataloader = get_dataloader(
            args.dataset,
            "validation",
            args.dev_batch_size,
            tokenizer,
            encoder,
            args.max_dialogue_len,
            args.max_turn_len,
            train_ratio=args.dataset_train_ratio,
            seed=args.seed,
        )
        torch.save(
            dev_dataloader, os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader")
        )
        test_dataloader = get_dataloader(
            args.dataset,
            "test",
            args.test_batch_size,
            tokenizer,
            encoder,
            args.max_dialogue_len,
            args.max_turn_len,
            train_ratio=args.dataset_train_ratio,
            seed=args.seed,
        )
        torch.save(
            test_dataloader, os.path.join(OUTPUT_DIR, "dataloaders", "test.dataloader")
        )

        setup_ensemble(OUTPUT_DIR, args.ensemble_size)

        logger.info(
            f"Building {args.ensemble_size} resampled dataloaders each of size {args.data_sampling_size}."
        )
        dataloaders = [
            dataloader_sample_dialogues(
                deepcopy(train_dataloader), args.data_sampling_size
            )
            for _ in tqdm(range(args.ensemble_size))
        ]
        logger.info("Dataloaders built.")

        for i, loader in enumerate(dataloaders):
            path = os.path.join(OUTPUT_DIR, "ens-%i" % i)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, "dataloaders", "train.dataloader")
            torch.save(loader, path)
        logger.info("Dataloaders saved.")

        # Do not perform standard training after ensemble setup is created
        return 0

    # Perform tasks
    # TRAINING
    if args.do_train:
        if os.path.exists(os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader")):
            train_dataloader = torch.load(
                os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader")
            )
            if train_dataloader.batch_size != args.train_batch_size:
                train_dataloader = change_batch_size(
                    train_dataloader, args.train_batch_size
                )
        else:
            if args.data_sampling_size <= 0:
                args.data_sampling_size = None
            if "distillation" not in config.loss_function:
                train_dataloader = get_dataloader(
                    args.dataset,
                    "train",
                    args.train_batch_size,
                    tokenizer,
                    encoder,
                    args.max_dialogue_len,
                    config.max_turn_len,
                    resampled_size=args.data_sampling_size,
                    train_ratio=args.dataset_train_ratio,
                    seed=args.seed,
                )
            else:
                reduction = "mean" if config.loss_function == "distillation" else "none"
                reduction = (
                    "dirichlet"
                    if config.loss_function == "distillation_bayesianmatching"
                    else reduction
                )
                loader_args = {
                    "ensemble_path": args.ensemble_model_path,
                    "set_type": "train",
                    "batch_size": args.train_batch_size,
                    "reduction": reduction,
                }
                train_dataloader = get_distillation_dataloader(**loader_args)
            torch.save(
                train_dataloader,
                os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader"),
            )

        # Get development set batch loaders= and ontology embeddings
        if args.do_eval:
            if os.path.exists(
                os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader")
            ):
                dev_dataloader = torch.load(
                    os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader")
                )
                if dev_dataloader.batch_size != args.dev_batch_size:
                    dev_dataloader = change_batch_size(
                        dev_dataloader, args.dev_batch_size
                    )
            else:
                if "distillation" not in config.loss_function:
                    dev_dataloader = get_dataloader(
                        args.dataset,
                        "validation",
                        args.dev_batch_size,
                        tokenizer,
                        encoder,
                        args.max_dialogue_len,
                        config.max_turn_len,
                    )
                else:
                    reduction = (
                        "mean" if config.loss_function == "distillation" else "none"
                    )
                    reduction = (
                        "dirichlet"
                        if config.loss_function == "distillation_bayesianmatching"
                        else reduction
                    )
                    loader_args = {
                        "ensemble_path": args.ensemble_model_path,
                        "set_type": "dev",
                        "batch_size": args.dev_batch_size,
                        "reduction": reduction,
                    }
                    dev_dataloader = get_distillation_dataloader(**loader_args)
                torch.save(
                    dev_dataloader,
                    os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader"),
                )
        else:
            dev_dataloader = None

        # TRAINING !!!!!!!!!!!!!!!!!!
        trainer = SetSUMBTTrainer(
            args,
            model,
            tokenizer,
            train_dataloader,
            dev_dataloader,
            logger,
            tb_writer,
            device,
        )
        trainer.train()

        # Copy final best model to the output dir
        checkpoints = os.listdir(OUTPUT_DIR)
        checkpoints = [p for p in checkpoints if "checkpoint" in p]
        checkpoints = sorted([int(p.split("-")[-1]) for p in checkpoints])
        best_checkpoint = os.path.join(OUTPUT_DIR, f"checkpoint-{checkpoints[-1]}")
        files = [
            "pytorch_model.bin",
            "config.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ]
        for file in files:
            copy(os.path.join(best_checkpoint, file), os.path.join(OUTPUT_DIR, file))

        # Load best model for evaluation
        tokenizer = Tokenizer.from_pretrained(OUTPUT_DIR)
        model = SetSumbtModel.from_pretrained(OUTPUT_DIR)
        model = model.to(device)

    # Evaluation on the training set
    if args.do_eval_trainset:
        if os.path.exists(os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader")):
            train_dataloader = torch.load(
                os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader")
            )
            if train_dataloader.batch_size != args.train_batch_size:
                train_dataloader = change_batch_size(
                    train_dataloader, args.train_batch_size
                )
        else:
            train_dataloader = get_dataloader(
                args.dataset,
                "train",
                args.train_batch_size,
                tokenizer,
                encoder,
                args.max_dialogue_len,
                config.max_turn_len,
            )
            torch.save(
                train_dataloader,
                os.path.join(OUTPUT_DIR, "dataloaders", "train.dataloader"),
            )

        # EVALUATION
        trainer = SetSUMBTTrainer(
            args, model, tokenizer, None, train_dataloader, logger, tb_writer, device
        )
        trainer.eval_mode(load_slots=True)

        if not os.path.exists(os.path.join(OUTPUT_DIR, "predictions")):
            os.mkdir(os.path.join(OUTPUT_DIR, "predictions"))
        save_pred_dist_path = os.path.join(OUTPUT_DIR, "predictions", "train.data")
        metrics = trainer.evaluate(
            save_pred_dist_path=save_pred_dist_path, draw_calibration_diagram=True
        )
        trainer.log_info(metrics, logging_stage="dev")

    # Evaluation on the development set
    if args.do_eval:
        if os.path.exists(os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader")):
            dev_dataloader = torch.load(
                os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader")
            )
            if dev_dataloader.batch_size != args.dev_batch_size:
                dev_dataloader = change_batch_size(dev_dataloader, args.dev_batch_size)
        else:
            dev_dataloader = get_dataloader(
                args.dataset,
                "validation",
                args.dev_batch_size,
                tokenizer,
                encoder,
                args.max_dialogue_len,
                config.max_turn_len,
            )
            torch.save(
                dev_dataloader,
                os.path.join(OUTPUT_DIR, "dataloaders", "dev.dataloader"),
            )

        # EVALUATION
        trainer = SetSUMBTTrainer(
            args, model, tokenizer, None, dev_dataloader, logger, tb_writer, device
        )
        trainer.eval_mode(load_slots=True)

        if not os.path.exists(os.path.join(OUTPUT_DIR, "predictions")):
            os.mkdir(os.path.join(OUTPUT_DIR, "predictions"))
        save_pred_dist_path = os.path.join(OUTPUT_DIR, "predictions", "dev.data")
        metrics = trainer.evaluate(
            save_eval_path=os.path.join(OUTPUT_DIR, "predictions", "dev.json"),
            save_pred_dist_path=save_pred_dist_path,
            draw_calibration_diagram=True,
        )
        trainer.log_info(metrics, logging_stage="dev", send_email=True)

    # Evaluation on the test set
    if args.do_test:
        if os.path.exists(os.path.join(OUTPUT_DIR, "dataloaders", "test.dataloader")):
            test_dataloader = torch.load(
                os.path.join(OUTPUT_DIR, "dataloaders", "test.dataloader")
            )
            if test_dataloader.batch_size != args.test_batch_size:
                test_dataloader = change_batch_size(
                    test_dataloader, args.test_batch_size
                )
        else:
            test_dataloader = get_dataloader(
                args.dataset,
                "test",
                args.test_batch_size,
                tokenizer,
                encoder,
                args.max_dialogue_len,
                config.max_turn_len,
            )
            torch.save(
                test_dataloader,
                os.path.join(OUTPUT_DIR, "dataloaders", "test.dataloader"),
            )

        trainer = SetSUMBTTrainer(
            args, model, tokenizer, None, test_dataloader, logger, tb_writer, device
        )
        trainer.eval_mode(load_slots=True)

        # TESTING
        if not os.path.exists(os.path.join(OUTPUT_DIR, "predictions")):
            os.mkdir(os.path.join(OUTPUT_DIR, "predictions"))

        save_pred_dist_path = os.path.join(OUTPUT_DIR, "predictions", "test.data")
        metrics = trainer.evaluate(
            save_eval_path=os.path.join(OUTPUT_DIR, "predictions", "test.json"),
            save_pred_dist_path=save_pred_dist_path,
            draw_calibration_diagram=True,
        )
        trainer.log_info(metrics, logging_stage="test", send_email=True)

        # Save final model for inference
        if not args.ensemble:
            trainer.model.save_pretrained(OUTPUT_DIR)
            trainer.tokenizer.save_pretrained(OUTPUT_DIR)

    tb_writer.close()


if __name__ == "__main__":
    main()
