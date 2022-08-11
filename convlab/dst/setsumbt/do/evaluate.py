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
"""Run SetSUMBT Calibration"""

import logging
import os

import torch
from transformers import (BertModel, BertConfig, BertTokenizer,
                          RobertaModel, RobertaConfig, RobertaTokenizer)

from convlab.dst.setsumbt.modeling import BertSetSUMBT, RobertaSetSUMBT
from convlab.dst.setsumbt.dataset import unified_format
from convlab.dst.setsumbt.dataset import ontology as embeddings
from convlab.dst.setsumbt.utils import get_args, update_args
from convlab.dst.setsumbt.modeling import evaluation_utils
from convlab.dst.setsumbt.loss.uncertainty_measures import ece, jg_ece, l2_acc
from convlab.dst.setsumbt.modeling import training


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
    args.output_dir = OUTPUT_DIR
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'predictions')):
        os.mkdir(os.path.join(OUTPUT_DIR, 'predictions'))

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

    # Create logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M %m-%d-%y')

    fh = logging.FileHandler(args.logging_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Get device
    if torch.cuda.is_available() and args.n_gpu > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        args.n_gpu = 0

    if args.n_gpu == 0:
        args.fp16 = False

    # Set up model training/evaluation
    evaluation_utils.set_seed(args)

    # Perform tasks
    if os.path.exists(os.path.join(OUTPUT_DIR, 'predictions', 'test.predictions')):
        pred = torch.load(os.path.join(OUTPUT_DIR, 'predictions', 'test.predictions'))
        state_labels = pred['state_labels']
        belief_states = pred['belief_states']
        if 'request_labels' in pred:
            request_labels = pred['request_labels']
            request_probs = pred['request_probs']
            active_domain_labels = pred['active_domain_labels']
            active_domain_probs = pred['active_domain_probs']
            general_act_labels = pred['general_act_labels']
            general_act_probs = pred['general_act_probs']
        else:
            request_probs = None
        del pred
    else:
        # Get training batch loaders and ontology embeddings
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader')):
            test_dataloader = torch.load(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader'))
            if test_dataloader.batch_size != args.test_batch_size:
                test_dataloader = unified_format.change_batch_size(test_dataloader, args.test_batch_size)
        else:
            tokenizer = Tokenizer(config.candidate_embedding_model_name)
            test_dataloader = unified_format.get_dataloader(args.dataset, 'test',
                                                            args.test_batch_size, tokenizer, args.max_dialogue_len,
                                                            config.max_turn_len)
            torch.save(test_dataloader, os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader'))

        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'test.db')):
            test_slots = torch.load(os.path.join(OUTPUT_DIR, 'database', 'test.db'))
        else:
            encoder = CandidateEncoderModel.from_pretrained(config.candidate_embedding_model_name)
            test_slots = embeddings.get_slot_candidate_embeddings(test_dataloader.dataset.ontology,
                                                                  'test', args, tokenizer, encoder)

        # Initialise Model
        model = SetSumbtModel.from_pretrained(args.model_name_or_path, config=config)
        model = model.to(device)

        training.set_ontology_embeddings(model, test_slots)

        belief_states = evaluation_utils.get_predictions(args, model, device, test_dataloader)
        state_labels = belief_states[1]
        request_probs = belief_states[2]
        request_labels = belief_states[3]
        active_domain_probs = belief_states[4]
        active_domain_labels = belief_states[5]
        general_act_probs = belief_states[6]
        general_act_labels = belief_states[7]
        belief_states = belief_states[0]
        out = {'belief_states': belief_states, 'state_labels': state_labels, 'request_probs': request_probs,
               'request_labels': request_labels, 'active_domain_probs': active_domain_probs,
               'active_domain_labels': active_domain_labels, 'general_act_probs': general_act_probs,
               'general_act_labels': general_act_labels}
        torch.save(out, os.path.join(OUTPUT_DIR, 'predictions', 'test.predictions'))

    # Calculate calibration metrics
    jg = jg_ece(belief_states, state_labels, 10)
    logger.info('Joint Goal ECE: %f' % jg)

    jg_acc = 0.0
    padding = torch.cat([item.unsqueeze(-1) for _, item in state_labels.items()], -1).sum(-1) * -1.0
    padding = (padding == len(state_labels))
    padding = padding.reshape(-1)
    for slot in belief_states:
        p_ = belief_states[slot]
        gold = state_labels[slot]

        pred = p_.reshape(-1, p_.size(-1)).argmax(dim=-1).unsqueeze(-1)
        acc = [lab in s for lab, s, pad in zip(gold.reshape(-1), pred, padding) if not pad]
        acc = torch.tensor(acc).float()

        jg_acc += acc

    n_turns = jg_acc.size(0)
    jg_acc = sum((jg_acc / len(belief_states)).int()).float()

    jg_acc /= n_turns

    logger.info(f'Joint Goal Accuracy: {jg_acc}')

    l2 = l2_acc(belief_states, state_labels, remove_belief=False)
    logger.info(f'Model L2 Norm Goal Accuracy: {l2}')
    l2 = l2_acc(belief_states, state_labels, remove_belief=True)
    logger.info(f'Binary Model L2 Norm Goal Accuracy: {l2}')

    padding = torch.cat([item.unsqueeze(-1) for _, item in state_labels.items()], -1).sum(-1) * -1.0
    padding = (padding == len(state_labels))
    padding = padding.reshape(-1)

    tp, fp, fn, tn, n = 0.0, 0.0, 0.0, 0.0, 0.0
    for slot in belief_states:
        p_ = belief_states[slot]
        gold = state_labels[slot].reshape(-1)
        p_ = p_.reshape(-1, p_.size(-1))

        p_ = p_[~padding].argmax(-1)
        gold = gold[~padding]

        tp += (p_ == gold)[gold != 0].int().sum().item()
        fp += (p_ != 0)[gold == 0].int().sum().item()
        fp += (p_ != gold)[gold != 0].int().sum().item()
        fp -= (p_ == 0)[gold != 0].int().sum().item()
        fn += (p_ == 0)[gold != 0].int().sum().item()
        tn += (p_ == 0)[gold == 0].int().sum().item()
        n += p_.size(0)

    acc = (tp + tn) / n
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)

    logger.info(f"Slot Accuracy: {acc}, Slot F1: {f1}, Slot Precision: {prec}, Slot Recall: {rec}")

    if request_probs is not None:
        tp, fp, fn = 0.0, 0.0, 0.0
        for slot in request_probs:
            p = request_probs[slot]
            l = request_labels[slot]

            tp += (p.round().int() * (l == 1)).reshape(-1).float()
            fp += (p.round().int() * (l == 0)).reshape(-1).float()
            fn += ((1 - p.round().int()) * (l == 1)).reshape(-1).float()
        tp /= len(request_probs)
        fp /= len(request_probs)
        fn /= len(request_probs)
        f1 = tp.sum() / (tp.sum() + 0.5 * (fp.sum() + fn.sum()))
        logger.info('Request F1 Score: %f' % f1.item())

        for slot in request_probs:
            p = request_probs[slot]
            p = p.unsqueeze(-1)
            p = torch.cat((1 - p, p), -1)
            request_probs[slot] = p
        jg = jg_ece(request_probs, request_labels, 10)
        logger.info('Request Joint Goal ECE: %f' % jg)

        tp, fp, fn = 0.0, 0.0, 0.0
        for dom in active_domain_probs:
            p = active_domain_probs[dom]
            l = active_domain_labels[dom]

            tp += (p.round().int() * (l == 1)).reshape(-1).float()
            fp += (p.round().int() * (l == 0)).reshape(-1).float()
            fn += ((1 - p.round().int()) * (l == 1)).reshape(-1).float()
        tp /= len(active_domain_probs)
        fp /= len(active_domain_probs)
        fn /= len(active_domain_probs)
        f1 = tp.sum() / (tp.sum() + 0.5 * (fp.sum() + fn.sum()))
        logger.info('Domain F1 Score: %f' % f1.item())

        for dom in active_domain_probs:
            p = active_domain_probs[dom]
            p = p.unsqueeze(-1)
            p = torch.cat((1 - p, p), -1)
            active_domain_probs[dom] = p
        jg = jg_ece(active_domain_probs, active_domain_labels, 10)
        logger.info('Domain Joint Goal ECE: %f' % jg)

        tp = ((general_act_probs.argmax(-1) > 0) *
              (general_act_labels > 0)).reshape(-1).float().sum()
        fp = ((general_act_probs.argmax(-1) > 0) *
              (general_act_labels == 0)).reshape(-1).float().sum()
        fn = ((general_act_probs.argmax(-1) == 0) *
              (general_act_labels > 0)).reshape(-1).float().sum()
        f1 = tp / (tp + 0.5 * (fp + fn))
        logger.info('General Act F1 Score: %f' % f1.item())

        err = ece(general_act_probs.reshape(-1, general_act_probs.size(-1)),
                  general_act_labels.reshape(-1), 10)
        logger.info('General Act ECE: %f' % err)

        for slot in request_probs:
            p = request_probs[slot].unsqueeze(-1)
            request_probs[slot] = torch.cat((1 - p, p), -1)

        l2 = l2_acc(request_probs, request_labels, remove_belief=False)
        logger.info(f'Model L2 Norm Request Accuracy: {l2}')
        l2 = l2_acc(request_probs, request_labels, remove_belief=True)
        logger.info(f'Binary Model L2 Norm Request Accuracy: {l2}')

        for slot in active_domain_probs:
            p = active_domain_probs[slot].unsqueeze(-1)
            active_domain_probs[slot] = torch.cat((1 - p, p), -1)

        l2 = l2_acc(active_domain_probs, active_domain_labels, remove_belief=False)
        logger.info(f'Model L2 Norm Domain Accuracy: {l2}')
        l2 = l2_acc(active_domain_probs, active_domain_labels, remove_belief=True)
        logger.info(f'Binary Model L2 Norm Domain Accuracy: {l2}')

        general_act_labels = {'general': general_act_labels}
        general_act_probs = {'general': general_act_probs}

        l2 = l2_acc(general_act_probs, general_act_labels, remove_belief=False)
        logger.info(f'Model L2 Norm General Act Accuracy: {l2}')
        l2 = l2_acc(general_act_probs, general_act_labels, remove_belief=False)
        logger.info(f'Binary Model L2 Norm General Act Accuracy: {l2}')


if __name__ == "__main__":
    main()
