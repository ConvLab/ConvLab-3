# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Discriminative models calibration"""

import random
import os

import torch
import numpy as np
from torch.distributions import Categorical
from torch.nn.functional import kl_div
from torch.nn import Module
from tqdm import tqdm


# Load logger and tensorboard summary writer
def set_logger(logger_, tb_writer_):
    global logger, tb_writer
    logger = logger_
    tb_writer = tb_writer_


# Set seeds
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    logger.info('Seed set to %d.' % args.seed)


def build_train_loaders(args, tokenizer, dataset):
    dataloaders = [dataset.get_dataloader('train', args.train_batch_size, tokenizer, args.max_dialogue_len,
                                            args.max_turn_len, resampled_size=args.data_sampling_size)
                        for _ in range(args.ensemble_size)]
    return dataloaders
