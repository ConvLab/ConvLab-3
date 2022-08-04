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
"""Get ensemble predictions and build distillation dataloaders"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from convlab.dst.setsumbt.dataset.unified_format import UnifiedFormatDataset
from convlab.dst.setsumbt.modeling import EnsembleSetSUMBT
from convlab.dst.setsumbt.modeling import training

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--set_type', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--reduction', type=str, default='mean')
    parser.add_argument('--get_ensemble_distributions', action='store_true')
    parser.add_argument('--build_dataloaders', action='store_true')
    args = parser.parse_args()

    if args.get_ensemble_distributions:
        get_ensemble_distributions(args)
    elif args.build_dataloaders:
        path = os.path.join(args.model_path, 'dataloaders', f'{args.set_type}.data')
        data = torch.load(path)

        reader = open(os.path.join(args.model_path, 'database', f'{args.set_type}.json'), 'r')
        ontology = json.load(reader)
        reader.close()

        loader = get_loader(data, ontology, args.set_type, args.batch_size)

        path = os.path.join(args.model_path, 'dataloaders', f'{args.set_type}.dataloader')
        torch.save(loader, path)
    else:
        raise NameError("NotImplemented")


def get_loader(data: dict, ontology: dict, set_type: str = 'train', batch_size: int = 3) -> DataLoader:
    """
    Build dataloader from ensemble prediction data

    Args:
        data: Dictionary of ensemble predictions
        ontology: Data ontology
        set_type: Data subset (train/validation/test)
        batch_size: Number of dialogues per batch

    Returns:
        loader: Data loader object
    """
    data = flatten_data(data)
    data = do_label_padding(data)
    data = UnifiedFormatDataset.from_datadict(data, ontology)
    if set_type == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return loader


def do_label_padding(data: dict) -> dict:
    """
    Add padding to the ensemble predictions (used as labels in distillation)

    Args:
        data: Dictionary of ensemble predictions

    Returns:
        data: Padded ensemble predictions
    """
    if 'attention_mask' in data:
        dialogs, turns = torch.where(data['attention_mask'].sum(-1) == 0.0)
    else:
        dialogs, turns = torch.where(data['input_ids'].sum(-1) == 0.0)
    
    for key in data:
        if key not in ['input_ids', 'attention_mask', 'token_type_ids']:
            data[key][dialogs, turns] = -1
    
    return data


map_dict = {'belief_state': 'belief', 'greeting_act_belief': 'goodbye_belief',
            'state_labels': 'labels', 'request_labels': 'request',
            'domain_labels': 'active', 'greeting_labels': 'goodbye'}


def flatten_data(data: dict) -> dict:
    """
    Map data to flattened feature format used in training
    Args:
        data: Ensemble prediction data

    Returns:
        data: Flattened ensemble prediction data
    """
    data_new = dict()
    for label, feats in data.items():
        label = map_dict.get(label, label)
        if type(feats) == dict:
            for label_, feats_ in feats.items():
                data_new[label + '-' + label_] = feats_
        else:
            data_new[label] = feats
    
    return data_new


def get_ensemble_distributions(args):
    """
    Load data and get ensemble predictions
    Args:
        args: Runtime arguments
    """
    device = DEVICE

    model = EnsembleSetSUMBT.from_pretrained(args.model_path)
    model = model.to(device)

    print('Model Loaded!')

    dataloader = os.path.join(args.model_path, 'dataloaders', f'{args.set_type}.dataloader')
    database = os.path.join(args.model_path, 'database', f'{args.set_type}.db')

    dataloader = torch.load(dataloader)
    database = torch.load(database)

    training.set_ontology_embeddings(model, database)

    print('Environment set up.')

    input_ids = []
    token_type_ids = []
    attention_mask = []
    state_labels = {slot: [] for slot in model.informable_slot_ids}
    request_labels = {slot: [] for slot in model.requestable_slot_ids}
    domain_labels = {domain: [] for domain in model.domain_ids}
    greeting_labels = []

    is_noisy = [] if 'is_noisy' in dataloader.dataset.features else None

    belief_state = {slot: [] for slot in model.informable_slot_ids}
    request_belief = {slot: [] for slot in model.requestable_slot_ids}
    domain_belief = {domain: [] for domain in model.domain_ids}
    greeting_act_belief = []
    model.eval()
    for batch in tqdm(dataloader, desc='Batch:'):
        ids = batch['input_ids']
        tt_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
        mask = batch['attention_mask'] if 'attention_mask' in batch else None

        if 'is_noisy' in batch:
            is_noisy.append(batch['is_noisy'])

        input_ids.append(ids)
        token_type_ids.append(tt_ids)
        attention_mask.append(mask)

        ids = ids.to(device)
        tt_ids = tt_ids.to(device) if tt_ids is not None else None
        mask = mask.to(device) if mask is not None else None

        for slot in state_labels:
            state_labels[slot].append(batch['labels-' + slot])
        if model.config.predict_actions:
            for slot in request_labels:
                request_labels[slot].append(batch['request-' + slot])
            for domain in domain_labels:
                domain_labels[domain].append(batch['active-' + domain])
            greeting_labels.append(batch['goodbye'])

        with torch.no_grad():
            p, p_req, p_dom, p_bye, _ = model(ids, mask, tt_ids,
                                            reduction=args.reduction)

        for slot in belief_state:
            belief_state[slot].append(p[slot].cpu())
        if model.config.predict_actions:
            for slot in request_belief:
                request_belief[slot].append(p_req[slot].cpu())
            for domain in domain_belief:
                domain_belief[domain].append(p_dom[domain].cpu())
            greeting_act_belief.append(p_bye.cpu())
    
    input_ids = torch.cat(input_ids, 0) if input_ids[0] is not None else None
    token_type_ids = torch.cat(token_type_ids, 0) if token_type_ids[0] is not None else None
    attention_mask = torch.cat(attention_mask, 0) if attention_mask[0] is not None else None
    is_noisy = torch.cat(is_noisy, 0) if is_noisy is not None else None

    state_labels = {slot: torch.cat(l, 0) for slot, l in state_labels.items()}
    if model.config.predict_actions:
        request_labels = {slot: torch.cat(l, 0) for slot, l in request_labels.items()}
        domain_labels = {domain: torch.cat(l, 0) for domain, l in domain_labels.items()}
        greeting_labels = torch.cat(greeting_labels, 0)
    
    belief_state = {slot: torch.cat(p, 0) for slot, p in belief_state.items()}
    if model.config.predict_actions:
        request_belief = {slot: torch.cat(p, 0) for slot, p in request_belief.items()}
        domain_belief = {domain: torch.cat(p, 0) for domain, p in domain_belief.items()}
        greeting_act_belief = torch.cat(greeting_act_belief, 0)

    data = {'input_ids': input_ids}
    if token_type_ids is not None:
        data['token_type_ids'] = token_type_ids
    if attention_mask is not None:
        data['attention_mask'] = attention_mask
    if is_noisy is not None:
        data['is_noisy'] = is_noisy
    data['state_labels'] = state_labels
    data['belief_state'] = belief_state
    if model.config.predict_actions:
        data['request_labels'] = request_labels
        data['domain_labels'] = domain_labels
        data['greeting_labels'] = greeting_labels
        data['request_belief'] = request_belief
        data['domain_belief'] = domain_belief
        data['greeting_act_belief'] = greeting_act_belief

    file = os.path.join(args.model_path, 'dataloaders', f'{args.set_type}.data')
    torch.save(data, file)


if __name__ == "__main__":
    main()
