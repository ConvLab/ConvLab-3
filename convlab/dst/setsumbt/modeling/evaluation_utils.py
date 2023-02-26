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


def set_seed(args):
    """
    Set random seeds

    Args:
        args (Arguments class): Arguments class containing seed and number of gpus to use
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_predictions(args, model, device: torch.device, dataloader: torch.utils.data.DataLoader) -> tuple:
    """
    Get model predictions

    Args:
        args: Runtime arguments
        model: SetSUMBT Model
        device: Torch device
        dataloader: Dataloader containing eval data
    """
    model.eval()
    
    belief_states = {slot: [] for slot in model.setsumbt.informable_slot_ids}
    request_probs = {slot: [] for slot in model.setsumbt.requestable_slot_ids}
    active_domain_probs = {dom: [] for dom in model.setsumbt.domain_ids}
    general_act_probs = []
    state_labels = {slot: [] for slot in model.setsumbt.informable_slot_ids}
    request_labels = {slot: [] for slot in model.setsumbt.requestable_slot_ids}
    active_domain_labels = {dom: [] for dom in model.setsumbt.domain_ids}
    general_act_labels = []
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():    
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

            p, p_req, p_dom, p_gen, _ = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)

            for slot in belief_states:
                p_ = p[slot]
                labs = batch['state_labels-' + slot].to(device)
                
                belief_states[slot].append(p_)
                state_labels[slot].append(labs)
            
            if p_req is not None:
                for slot in request_probs:
                    p_ = p_req[slot]
                    labs = batch['request_labels-' + slot].to(device)

                    request_probs[slot].append(p_)
                    request_labels[slot].append(labs)
                
                for domain in active_domain_probs:
                    p_ = p_dom[domain]
                    labs = batch['active_domain_labels-' + domain].to(device)

                    active_domain_probs[domain].append(p_)
                    active_domain_labels[domain].append(labs)
                
                general_act_probs.append(p_gen)
                general_act_labels.append(batch['general_act_labels'].to(device))
    
    for slot in belief_states:
        belief_states[slot] = torch.cat(belief_states[slot], 0)
        state_labels[slot] = torch.cat(state_labels[slot], 0)
    if p_req is not None:
        for slot in request_probs:
            request_probs[slot] = torch.cat(request_probs[slot], 0)
            request_labels[slot] = torch.cat(request_labels[slot], 0)
        for domain in active_domain_probs:
            active_domain_probs[domain] = torch.cat(active_domain_probs[domain], 0)
            active_domain_labels[domain] = torch.cat(active_domain_labels[domain], 0)
        general_act_probs = torch.cat(general_act_probs, 0)
        general_act_labels = torch.cat(general_act_labels, 0)
    else:
        request_probs, request_labels, active_domain_probs, active_domain_labels = [None] * 4
        general_act_probs, general_act_labels = [None] * 2

    out = (belief_states, state_labels, request_probs, request_labels)
    out += (active_domain_probs, active_domain_labels, general_act_probs, general_act_labels)
    return out
