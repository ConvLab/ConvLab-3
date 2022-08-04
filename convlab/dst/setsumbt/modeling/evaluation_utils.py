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
"""Evaluation Utilities"""

import random

import torch
import numpy as np
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


def get_predictions(args, model, device, dataloader):
    logger.info("  Num Batches = %d", len(dataloader))

    model.eval()
    if args.dropout_iterations > 1:
        model.train()
    
    belief_states = {slot: [] for slot in model.informable_slot_ids}
    request_belief = {slot: [] for slot in model.requestable_slot_ids}
    domain_belief = {dom: [] for dom in model.domain_ids}
    greeting_belief = []
    labels = {slot: [] for slot in model.informable_slot_ids}
    request_labels = {slot: [] for slot in model.requestable_slot_ids}
    domain_labels = {dom: [] for dom in model.domain_ids}
    greeting_labels = []
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():    
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

            if args.dropout_iterations > 1:
                p = {slot: [] for slot in model.informable_slot_ids}
                for _ in range(args.dropout_iterations):
                    p_, p_req_, p_dom_, p_bye_, _ = model(input_ids=input_ids,
                                                        token_type_ids=token_type_ids,
                                                        attention_mask=attention_mask)
                    for slot in model.informable_slot_ids:
                        p[slot].append(p_[slot].unsqueeze(0))
                
                mu = {slot: torch.cat(p[slot], 0).mean(0) for slot in model.informable_slot_ids}
                sig = {slot: torch.cat(p[slot], 0).var(0) for slot in model.informable_slot_ids}
                p = {slot: mu[slot] / torch.sqrt(1 + sig[slot]) for slot in model.informable_slot_ids}
                p = {slot: normalise(p[slot]) for slot in model.informable_slot_ids}
            else:
                p, p_req, p_dom, p_bye, _ = model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
            
            for slot in model.informable_slot_ids:
                p_ = p[slot]
                labs = batch['labels-' + slot].to(device)
                
                belief_states[slot].append(p_)
                labels[slot].append(labs)
            
            if p_req is not None:
                for slot in model.requestable_slot_ids:
                    p_ = p_req[slot]
                    labs = batch['request-' + slot].to(device)

                    request_belief[slot].append(p_)
                    request_labels[slot].append(labs)
                
                for domain in model.domain_ids:
                    p_ = p_dom[domain]
                    labs = batch['active-' + domain].to(device)

                    domain_belief[domain].append(p_)
                    domain_labels[domain].append(labs)
                
                greeting_belief.append(p_bye)
                greeting_labels.append(batch['goodbye'].to(device))
    
    for slot in belief_states:
        belief_states[slot] = torch.cat(belief_states[slot], 0)
        labels[slot] = torch.cat(labels[slot], 0)
    if p_req is not None:
        for slot in request_belief:
            request_belief[slot] = torch.cat(request_belief[slot], 0)
            request_labels[slot] = torch.cat(request_labels[slot], 0)
        for domain in domain_belief:
            domain_belief[domain] = torch.cat(domain_belief[domain], 0)
            domain_labels[domain] = torch.cat(domain_labels[domain], 0)
        greeting_belief = torch.cat(greeting_belief, 0)
        greeting_labels = torch.cat(greeting_labels, 0)
    else:
        request_belief, request_labels, domain_belief, domain_labels, greeting_belief, greeting_labels = [None]*6

    out = (belief_states, labels, request_belief, request_labels)
    out += (domain_belief, domain_labels, greeting_belief, greeting_labels)
    return out


def normalise(p):
    p_shape = p.size()

    p = p.reshape(-1, p_shape[-1]) + 1e-10
    p_sum = p.sum(-1).unsqueeze(1).repeat((1, p_shape[-1]))
    p /= p_sum

    p = p.reshape(p_shape)

    return p
