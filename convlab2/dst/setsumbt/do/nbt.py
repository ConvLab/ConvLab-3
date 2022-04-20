# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Run SetSUMBT training/eval"""

import logging
import random
import os
from shutil import copy2 as copy

import torch
import transformers
from transformers import (BertModel, BertConfig, BertTokenizer,
                          RobertaModel, RobertaConfig, RobertaTokenizer)
from tensorboardX import SummaryWriter

from convlab2.dst.setsumbt.modeling.bert_nbt import BertSetSUMBT
from convlab2.dst.setsumbt.modeling.roberta_nbt import RobertaSetSUMBT
from convlab2.dst.setsumbt.unified_format_data import unified_format
from convlab2.dst.setsumbt.modeling import training
from convlab2.dst.setsumbt.unified_format_data import ontology as embeddings
from convlab2.dst.setsumbt.utils import get_args, update_args
# from convlab2.dst.setsumbt.modeling import ensemble_utils

# Available model
MODELS = {
    'bert': (BertSetSUMBT, BertModel, BertConfig, BertTokenizer),
    'roberta': (RobertaSetSUMBT, RobertaModel, RobertaConfig, RobertaTokenizer)
}


def main(args=None, config=None):
    # Get arguments
    if args is None:
        args, config = get_args(MODELS)

    if args.model_type in MODELS:
        SetSumbtModel, CandidateEncoderModel, ConfigClass, Tokenizer = MODELS[args.model_type]
    else:
        raise NameError('NotImplemented')

    # Set up output directory
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        os.mkdir(os.path.join(OUTPUT_DIR, 'database'))
        os.mkdir(os.path.join(OUTPUT_DIR, 'dataloaders'))
    args.output_dir = OUTPUT_DIR

    # Set pretrained model path to the trained checkpoint
    paths = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
    if 'pytorch_model.bin' in paths and 'config.json' in paths:
        args.model_name_or_path = args.output_dir
        config = ConfigClass.from_pretrained(args.model_name_or_path)
    else:
        paths = [os.path.join(args.output_dir, p) for p in paths if 'checkpoint-' in p]
        if paths:
            paths = paths[0]
            args.model_name_or_path = paths
            config = ConfigClass.from_pretrained(args.model_name_or_path)

    args = update_args(args, config)

    # Create TensorboardX writer
    tb_writer = SummaryWriter(logdir=args.tensorboard_path)

    # Create logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level=logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Get device
    if torch.cuda.is_available() and args.n_gpu > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        args.n_gpu = 0

    if args.n_gpu == 0:
        args.fp16 = False

    # Initialise Model
    model = SetSumbtModel.from_pretrained(args.model_name_or_path, config=config)
    model = model.to(device)

    # Create Tokenizer and embedding model for Data Loaders and ontology
    encoder = model.roberta if args.model_type == 'roberta' else None
    encoder = model.bert if args.model_type == 'bert' else encoder

    tokenizer = Tokenizer.from_pretrained(config.tokenizer_name, config=config)

    # Set up model training/evaluation
    training.set_logger(logger, tb_writer)
    training.set_seed(args)
    embeddings.set_seed(args)

    # if args.ensemble_size > 1:
    #     ensemble_utils.set_logger(logger, tb_writer)
    #     ensemble.set_seed(args)
    #     logger.info('Building %i resampled dataloaders each of size %i' % (args.ensemble_size,
    #                                                                        args.data_sampling_size))
    #     dataloaders = ensemble_utils.build_train_loaders(args, tokenizer, Dataset)
    #     logger.info('Dataloaders built.')
    #     for i, loader in enumerate(dataloaders):
    #         path = os.path.join(OUTPUT_DIR, 'ensemble-%i' % i)
    #         if not os.path.exists(path):
    #             os.mkdir(path)
    #         path = os.path.join(path, 'train.dataloader')
    #         torch.save(loader, path)
    #     logger.info('Dataloaders saved.')
    #
    #     train_slots = embeddings.get_slot_candidate_embeddings(
    #         'train', args, tokenizer, encoder)
    #     dev_slots = embeddings.get_slot_candidate_embeddings(
    #         'dev', args, tokenizer, encoder)
    #     test_slots = embeddings.get_slot_candidate_embeddings(
    #         'test', args, tokenizer, encoder)
    #
    #     train_dataloader = Dataset.get_dataloader(
    #         'train', args.train_batch_size, tokenizer, args.max_dialogue_len, config.max_turn_len)
    #     torch.save(dev_dataloader, os.path.join(
    #         OUTPUT_DIR, 'dataloaders', 'train.dataloader'))
    #     dev_dataloader = Dataset.get_dataloader(
    #         'dev', args.dev_batch_size, tokenizer, args.max_dialogue_len, config.max_turn_len)
    #     torch.save(dev_dataloader, os.path.join(
    #         OUTPUT_DIR, 'dataloaders', 'dev.dataloader'))
    #     test_dataloader = Dataset.get_dataloader(
    #         'test', args.test_batch_size, tokenizer, args.max_dialogue_len, config.max_turn_len)
    #     torch.save(test_dataloader, os.path.join(
    #         OUTPUT_DIR, 'dataloaders', 'test.dataloader'))
    #
    #     # Do not perform standard training after ensemble setup is created
    #     return 0

    # Perform tasks
    # TRAINING
    if args.do_train:
        exists = False
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'train.dataloader')):
            train_dataloader = torch.load(os.path.join(OUTPUT_DIR, 'dataloaders', 'train.dataloader'))
            if train_dataloader.batch_size == args.train_batch_size:
                exists = True
        if not exists:
            if args.data_sampling_size <= 0:
                args.data_sampling_size = None
            train_dataloader = unified_format.get_dataloader(args.dataset, 'train',
                                                args.train_batch_size, tokenizer, args.max_dialogue_len,
                                                config.max_turn_len, resampled_size=args.data_sampling_size)
            torch.save(train_dataloader, os.path.join(OUTPUT_DIR, 'dataloaders', 'train.dataloader'))

        # Get training batch loaders and ontology embeddings
        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'train.db')):
            train_slots = torch.load(os.path.join(OUTPUT_DIR, 'database', 'train.db'))
        else:
            train_slots = embeddings.get_slot_candidate_embeddings(train_dataloader.dataset.ontology,
                                                                   'train', args, tokenizer, encoder)

        # Get development set batch loaders= and ontology embeddings
        if args.do_eval:
            exists = False
            if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader')):
                dev_dataloader = torch.load(os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader'))
                if dev_dataloader.batch_size == args.dev_batch_size:
                    exists = True
            if not exists:
                dev_dataloader = unified_format.get_dataloader(args.dataset, 'validation',
                                                args.dev_batch_size, tokenizer, args.max_dialogue_len,
                                                config.max_turn_len)
                torch.save(dev_dataloader, os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader'))

            if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'dev.db')):
                dev_slots = torch.load(os.path.join(OUTPUT_DIR, 'database', 'dev.db'))
            else:
                dev_slots = embeddings.get_slot_candidate_embeddings(dev_dataloader.dataset.ontology,
                                                                     'dev', args, tokenizer, encoder)
        else:
            dev_dataloader = None
            dev_slots = None

        # Load model ontology
        training.set_ontology_embeddings(model, train_slots)

        # TRAINING !!!!!!!!!!!!!!!!!!
        training.train(args, model, device, train_dataloader, dev_dataloader, train_slots, dev_slots,
                       embeddings=embeddings, tokenizer=tokenizer)

        # Copy final best model to the output dir
        checkpoints = os.listdir(OUTPUT_DIR)
        checkpoints = [p for p in checkpoints if 'checkpoint' in p]
        checkpoints = sorted([int(p.split('-')[-1]) for p in checkpoints])
        best_checkpoint = checkpoints[-1]
        best_checkpoint = os.path.join(
            OUTPUT_DIR, f'checkpoint-{best_checkpoint}')
        copy(os.path.join(best_checkpoint, 'pytorch_model.bin'),
             os.path.join(OUTPUT_DIR, 'pytorch_model.bin'))
        copy(os.path.join(best_checkpoint, 'config.json'),
             os.path.join(OUTPUT_DIR, 'config.json'))

        # Load best model for evaluation
        model = SetSumbtModel.from_pretrained(OUTPUT_DIR)
        model = model.to(device)

    # Evaluation on the development set
    if args.do_eval:
        exists = False
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader')):
            dev_dataloader = torch.load(os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader'))
            if dev_dataloader.batch_size == args.dev_batch_size:
                exists = True
        if not exists:
            dev_dataloader = unified_format.get_dataloader(args.dataset, 'validation',
                                                           args.dev_batch_size, tokenizer, args.max_dialogue_len,
                                                           config.max_turn_len)
            torch.save(dev_dataloader, os.path.join(OUTPUT_DIR, 'dataloaders', 'dev.dataloader'))

        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'dev.db')):
            dev_slots = torch.load(os.path.join(OUTPUT_DIR, 'database', 'dev.db'))
        else:
            dev_slots = embeddings.get_slot_candidate_embeddings(dev_dataloader.dataset.ontology,
                                                                 'dev', args, tokenizer, encoder)

        # Load model ontology
        training.set_ontology_embeddings(model, dev_slots)

        # EVALUATION
        jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss = training.evaluate(args, model, device, dev_dataloader)
        if req_f1:
            logger.info('Development loss: %f, Joint Goal Accuracy: %f, Slot Accuracy: %f, Request F1 Score: %f, Domain F1 Score: %f, Goodbye F1 Score: %f'
                        % (loss, jg_acc, sl_acc, req_f1, dom_f1, bye_f1))
        else:
            logger.info('Development loss: %f, Joint Goal Accuracy: %f, Slot Accuracy: %f'
                        % (loss, jg_acc, sl_acc))

    # Evaluation on the test set
    if args.do_test:
        exists = False
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader')):
            test_dataloader = torch.load(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader'))
            if test_dataloader.batch_size == args.test_batch_size:
                exists = True
        if not exists:
            test_dataloader = unified_format.get_dataloader(args.dataset, 'test',
                                                           args.test_batch_size, tokenizer, args.max_dialogue_len,
                                                           config.max_turn_len)
            torch.save(test_dataloader, os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader'))

        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'test.db')):
            test_slots = torch.load(os.path.join(OUTPUT_DIR, 'database', 'test.db'))
        else:
            test_slots = embeddings.get_slot_candidate_embeddings(test_dataloader.dataset.ontology,
                                                                 'test', args, tokenizer, encoder)

        # Load model ontology
        training.set_ontology_embeddings(model, test_slots)

        # TESTING
        jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss = training.evaluate(args, model, device, test_dataloader)
        if req_f1:
            logger.info('Test loss: %f, Joint Goal Accuracy: %f, Slot Accuracy: %f, Request F1 Score: %f, Domain F1 Score: %f, Goodbye F1 Score: %f'
                        % (loss, jg_acc, sl_acc, req_f1, dom_f1, bye_f1))
        else:
            logger.info('Test loss: %f, Joint Goal Accuracy: %f, Slot Accuracy: %f'
                        % (loss, jg_acc, sl_acc))

    tb_writer.close()


if __name__ == "__main__":
    main()
