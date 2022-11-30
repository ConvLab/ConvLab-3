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
"""SetSUMBT utils"""

import os
import json
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime

from git import Repo


def get_args(base_models: dict):
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Config file usage
    parser.add_argument('--starting_config_name', default=None, type=str)

    # Optional
    parser.add_argument('--tensorboard_path', help='Path to tensorboard', default='')
    parser.add_argument('--logging_path', help='Path for log file', default='')
    parser.add_argument('--seed', help='Seed value for reproducibility', default=0, type=int)

    # DATASET (Optional)
    parser.add_argument('--dataset', help='Dataset Name (See Convlab 3 unified format for possible datasets',
                        default='multiwoz21')
    parser.add_argument('--dataset_train_ratio', help='Fraction of training set to use in training', default=1.0,
                        type=float)
    parser.add_argument('--max_dialogue_len', help='Maximum number of turns per dialogue', default=12, type=int)
    parser.add_argument('--max_turn_len', help='Maximum number of tokens per turn', default=64, type=int)
    parser.add_argument('--max_slot_len', help='Maximum number of tokens per slot description', default=12, type=int)
    parser.add_argument('--max_candidate_len', help='Maximum number of tokens per value candidate', default=12,
                        type=int)
    parser.add_argument('--force_processing', action='store_true', help='Force preprocessing of data.')
    parser.add_argument('--data_sampling_size', help='Resampled dataset size', default=-1, type=int)
    parser.add_argument('--no_descriptions', help='Do not use slot descriptions rather than slot names for embeddings',
                        action='store_true')

    # MODEL
    # Environment
    parser.add_argument('--output_dir', help='Output storage directory', default=None)
    parser.add_argument('--model_type', help='Encoder Model Type: bert/roberta', default='roberta')
    parser.add_argument('--model_name_or_path', help='Name or path of the pretrained model.', default=None)
    parser.add_argument('--candidate_embedding_model_name', default=None,
                        help='Name of the pretrained candidate embedding model.')
    parser.add_argument('--transformers_local_files_only', help='Use local files only for huggingface transformers',
                        action='store_true')

    # Architecture
    parser.add_argument('--freeze_encoder', help='No training performed on the turn encoder Bert Model',
                        action='store_true')
    parser.add_argument('--slot_attention_heads', help='Number of attention heads for slot conditioning',
                        default=12, type=int)
    parser.add_argument('--dropout_rate', help='Dropout Rate', default=0.3, type=float)
    parser.add_argument('--nbt_type', help='Belief Tracker type: gru/lstm', default='gru')
    parser.add_argument('--nbt_hidden_size', help='Hidden embedding size for the Neural Belief Tracker',
                        default=300, type=int)
    parser.add_argument('--nbt_layers', help='Number of RNN layers in the NBT', default=1, type=int)
    parser.add_argument('--rnn_zero_init', help='Zero Initialise RNN hidden states', action='store_true')
    parser.add_argument('--distance_measure', default='cosine',
                        help='Similarity measure for candidate scoring: cosine/euclidean')
    parser.add_argument('--ensemble_size', help='Number of models in ensemble', default=-1, type=int)
    parser.add_argument('--no_set_similarity', action='store_true', help='Set True to not use set similarity')
    parser.add_argument('--set_pooling',
                        help='Set pooling method for set similarity model using single embedding distances',
                        default='cnn')
    parser.add_argument('--candidate_pooling',
                        help='Pooling approach for non set based candidate representations: cls/mean',
                        default='mean')
    parser.add_argument('--no_action_prediction', help='Model does not predicts user actions and active domain',
                        action='store_true')

    # Loss
    parser.add_argument('--loss_function',
                        help='Loss Function for training: crossentropy/bayesianmatching/labelsmoothing/...',
                        default='labelsmoothing')
    parser.add_argument('--kl_scaling_factor', help='Scaling factor for KL divergence in bayesian matching loss',
                        type=float)
    parser.add_argument('--prior_constant', help='Constant parameter for prior in bayesian matching loss',
                        type=float)
    parser.add_argument('--ensemble_smoothing', help='Ensemble distribution smoothing constant', type=float)
    parser.add_argument('--annealing_base_temp', help='Ensemble Distribution distillation temp annealing base temp',
                        type=float)
    parser.add_argument('--annealing_cycle_len', help='Ensemble Distribution distillation temp annealing cycle length',
                        type=float)
    parser.add_argument('--label_smoothing', help='Label smoothing coefficient.', type=float)
    parser.add_argument('--user_goal_loss_weight', help='Weight of the user goal prediction loss. 0.0<weight<=1.0',
                        type=float)
    parser.add_argument('--user_request_loss_weight',
                        help='Weight of the user request prediction loss. 0.0<weight<=1.0', type=float)
    parser.add_argument('--user_general_act_loss_weight',
                        help='Weight of the user general act prediction loss. 0.0<weight<=1.0', type=float)
    parser.add_argument('--active_domain_loss_weight',
                        help='Weight of the active domain prediction loss. 0.0<weight<=1.0', type=float)

    # TRAINING
    parser.add_argument('--train_batch_size', help='Training Set Batch Size', default=8, type=int)
    parser.add_argument('--max_training_steps', help='Maximum number of training update steps', default=-1, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of batches accumulated for one update step')
    parser.add_argument('--num_train_epochs', help='Number of training epochs', default=50, type=int)
    parser.add_argument('--patience', help='Number of training steps without improving model before stopping.',
                        default=20, type=int)
    parser.add_argument('--weight_decay', help='Weight decay rate', default=0.01, type=float)
    parser.add_argument('--learning_rate', help='Initial Learning Rate', default=5e-5, type=float)
    parser.add_argument('--warmup_proportion', help='Warmup proportion for linear scheduler', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', help='Maximum norm of the loss gradients', default=1.0, type=float)
    parser.add_argument('--save_steps', help='Number of update steps between saving model', default=-1, type=int)
    parser.add_argument('--keep_models', help='How many model checkpoints should be kept during training',
                        default=1, type=int)

    # CALIBRATION
    parser.add_argument('--temp_scaling', help='Temperature scaling coefficient', default=1.0, type=float)

    # EVALUATION
    parser.add_argument('--dev_batch_size', help='Dev Set Batch Size', default=16, type=int)
    parser.add_argument('--test_batch_size', help='Test Set Batch Size', default=16, type=int)

    # COMPUTING
    parser.add_argument('--n_gpu', help='Number of GPUs to use', default=1, type=int)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # ACTIONS
    parser.add_argument('--run_nbt', help='Run NBT script', action='store_true')
    parser.add_argument('--run_evaluation', help='Run evaluation script', action='store_true')

    # RUN_NBT ACTIONS
    parser.add_argument('--do_train', help='Perform training', action='store_true')
    parser.add_argument('--do_eval', help='Perform model evaluation during training', action='store_true')
    parser.add_argument('--do_test', help='Evaluate model on test data', action='store_true')
    args = parser.parse_args()

    if args.starting_config_name:
        args = get_starting_config(args)

    if args.do_train:
        args.do_eval = True

    # Simplify args
    args.set_similarity = not args.no_set_similarity
    args.use_descriptions = not args.no_descriptions
    args.predict_actions = not args.no_action_prediction

    # Setup default directories
    if not args.output_dir:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(args.output_dir, 'models')

        name = 'SetSUMBT' if args.set_similarity else 'SUMBT'
        name += '+ActPrediction' if args.predict_actions else ''
        name += '-' + args.dataset
        name += '-' + str(round(args.dataset_train_ratio*100)) + '%' if args.dataset_train_ratio != 1.0 else ''
        name += '-' + args.model_type
        name += '-' + args.nbt_type
        name += '-' + args.distance_measure
        name += '-' + args.loss_function
        name += '-Seed' + str(args.seed)
        name += datetime.now().strftime("-%d-%m-%y-%H-%M")

        args.output_dir = os.path.join(args.output_dir, name)

    # Default Loss funtions options
    if args.loss_function == 'bayesianmatching':
        if not args.kl_scaling_factor:
            args.kl_scaling_factor = 0.001
        if not args.prior_constant:
            args.prior_constant = 1.0
    if args.loss_function == 'labelsmoothing':
        if not args.label_smoothing:
            args.label_smoothing = 0.05
    if args.loss_function == 'distillation':
        if not args.ensemble_smoothing:
            args.ensemble_smoothing = 1e-4
        if not args.annealing_base_temp:
            args.annealing_base_temp = 2.5
        if not args.annealing_base_temp:
            args.annealing_cycle_len = 0.002

    # Default user action loss weight
    if args.predict_actions:
        if not args.user_goal_loss_weight:
            args.user_goal_loss_weight = 1.0
        if not args.user_request_loss_weight:
            args.user_request_loss_weight = 0.2
        if not args.user_general_act_loss_weight:
            args.user_general_act_loss_weight = 0.2
        if not args.active_domain_loss_weight:
            args.active_domain_loss_weight = 0.2

    args.tensorboard_path = args.tensorboard_path if args.tensorboard_path else os.path.join(args.output_dir, 'tb_logs')
    args.logging_path = args.logging_path if args.logging_path else os.path.join(args.output_dir, 'run.log')

    # Default model_name's
    if not args.model_name_or_path:
        if args.model_type == 'roberta':
            args.model_name_or_path = 'roberta-base'
        elif args.model_type == 'bert':
            args.model_name_or_path = 'bert-base-uncased'
        else:
            raise NameError('ModelNameNotSpecified')

    if not args.candidate_embedding_model_name:
        args.candidate_embedding_model_name = args.model_name_or_path

    if args.model_type in base_models:
        config_class = base_models[args.model_type][-2]
    else:
        raise NameError('NotImplemented')
    config = build_config(config_class, args)
    return args, config


def get_starting_config(args):
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, 'configs', f"{args.starting_config_name}.json")
    reader = open(path, 'r')
    config = json.load(reader)
    reader.close()

    if "model_type" in config:
        if config["model_type"].lower() == 'setsumbt':
            config["model_type"] = 'roberta'
            config["no_set_similarity"] = False
            config["no_descriptions"] = False
        elif config["model_type"].lower() == 'sumbt':
            config["model_type"] = 'bert'
            config["no_set_similarity"] = True
            config["no_descriptions"] = False

    variables = vars(args).keys()
    for key, value in config.items():
        if key in variables:
            setattr(args, key, value)

    return args


def get_git_info():
    repo = Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
    branch_name = repo.active_branch.name
    commit_hex = repo.head.object.hexsha

    info = f"{branch_name}/{commit_hex}"
    return info


def build_config(config_class, args):
    config = config_class.from_pretrained(args.model_name_or_path)
    config.code_version = get_git_info()
    if not os.path.exists(args.model_name_or_path):
        config.tokenizer_name = args.model_name_or_path
    try:
        config.tokenizer_name = config.tokenizer_name
    except AttributeError:
        config.tokenizer_name = args.model_name_or_path
    try:
        config.candidate_embedding_model_name = config.candidate_embedding_model_name
    except:
        if args.candidate_embedding_model_name:
            config.candidate_embedding_model_name = args.candidate_embedding_model_name
    config.max_dialogue_len = args.max_dialogue_len
    config.max_turn_len = args.max_turn_len
    config.max_slot_len = args.max_slot_len
    config.max_candidate_len = args.max_candidate_len
    config.use_descriptions = args.use_descriptions
    config.train_batch_size = args.train_batch_size
    config.dev_batch_size = args.dev_batch_size
    config.test_batch_size = args.test_batch_size
    config.seed = args.seed
    config.freeze_encoder = args.freeze_encoder
    config.slot_attention_heads = args.slot_attention_heads
    config.dropout_rate = args.dropout_rate
    config.nbt_type = args.nbt_type
    config.nbt_hidden_size = args.nbt_hidden_size
    config.nbt_layers = args.nbt_layers
    config.rnn_zero_init = args.rnn_zero_init
    config.distance_measure = args.distance_measure
    config.set_similarity = args.set_similarity
    config.predict_actions = args.predict_actions
    if config.set_similarity:
        config.set_pooling = args.set_pooling
    config.loss_function = args.loss_function
    if config.loss_function == 'bayesianmatching':
        config.kl_scaling_factor = args.kl_scaling_factor
        config.prior_constant = args.prior_constant
    if config.loss_function == 'inhibitedce':
        config.inhibiting_factor = args.inhibiting_factor
    if config.loss_function == 'labelsmoothing':
        config.label_smoothing = args.label_smoothing
    if config.loss_function == 'distillation':
        config.ensemble_smoothing = args.ensemble_smoothing
    if not config.set_similarity:
        config.candidate_pooling = args.candidate_pooling
    if args.predict_actions:
        config.user_goal_loss_weight = args.user_goal_loss_weight
        config.user_request_loss_weight = args.user_request_loss_weight
        config.user_general_act_loss_weight = args.user_general_act_loss_weight
        config.active_domain_loss_weight = args.active_domain_loss_weight

    return config


def update_args(args, config):
    try:
        args.candidate_embedding_model_name = config.candidate_embedding_model_name
    except AttributeError:
        args.candidate_embedding_model_name = None
    args.max_dialogue_len = config.max_dialogue_len
    args.max_turn_len = config.max_turn_len
    args.max_slot_len = config.max_slot_len
    args.max_candidate_len = config.max_candidate_len
    args.set_similarity = config.set_similarity
    args.predict_actions = config.predict_actions
    args.use_descriptions = config.use_descriptions
    args.predict_actions = config.predict_actions
    args.loss_function = config.loss_function
    args.seed = config.seed

    if not config.set_similarity:
        args.candidate_pooling = config.candidate_pooling

    return args


def clear_checkpoints(path, topn=1):
    checkpoints = os.listdir(path)
    checkpoints = [p for p in checkpoints if 'checkpoint' in p]
    checkpoints = sorted([int(p.split('-')[-1]) for p in checkpoints])
    checkpoints = checkpoints[:-topn]
    checkpoints = ['checkpoint-%i' % p for p in checkpoints]
    checkpoints = [os.path.join(path, p) for p in checkpoints]
    [shutil.rmtree(p) for p in checkpoints]
