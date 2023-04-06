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
"""Get ensemble predictions and build distillation dataloaders"""

import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from convlab.dst.setsumbt.datasets.unified_format import UnifiedFormatDataset
from convlab.dst.setsumbt.datasets.utils import IdTensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader(ensemble_path:str, set_type: str = 'train', batch_size: int = 3,
                   reduction: str = 'none') -> DataLoader:
    """
    Get dataloader for distillation of ensemble.

    Args:
        ensemble_path: Path to ensemble model and predictive distributions.
        set_type: Dataset split to load.
        batch_size: Batch size.
        reduction: Reduction to apply to ensemble predictive distributions.

    Returns:
        loader: Dataloader for distillation.
    """
    # Load data and predictions from ensemble
    path = os.path.join(ensemble_path, 'dataloaders', f"{set_type}.dataloader")
    dataset = torch.load(path).dataset

    path = os.path.join(ensemble_path, 'predictions', f"{set_type}.data")
    data = torch.load(path)

    dialogue_ids = data.pop('dialogue_ids')

    # Preprocess data
    data = reduce_data(data, reduction=reduction)
    data = flatten_data(data)
    data = do_label_padding(data)

    # Build dataset and dataloader
    data = UnifiedFormatDataset.from_datadict(set_type=set_type if set_type != 'dev' else 'validation',
                                              data=data,
                                              ontology=dataset.ontology,
                                              ontology_embeddings=dataset.ontology_embeddings)
    data.features['dialogue_ids'] = IdTensor(dialogue_ids)

    if set_type == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return loader


def reduce_data(data: dict, reduction: str = 'none') -> dict:
    """
    Reduce ensemble predictive distributions.

    Args:
        data: Dictionary of ensemble predictive distributions.
        reduction: Reduction to apply to ensemble predictive distributions.

    Returns:
        data: Reduced ensemble predictive distributions.
    """
    if reduction == 'mean':
        data['belief_state'] = {slot: probs.mean(-2) for slot, probs in data['belief_state'].items()}
        if 'request_probabilities' in data:
            data['request_probabilities'] = {slot: probs.mean(-1)
                                             for slot, probs in data['request_probabilities'].items()}
            data['active_domain_probabilities'] = {domain: probs.mean(-1)
                                                   for domain, probs in data['active_domain_probabilities'].items()}
            data['general_act_probabilities'] = data['general_act_probabilities'].mean(-2)
    return data


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
        if type(feats) == dict:
            for label_, feats_ in feats.items():
                data_new[label + '-' + label_] = feats_
        else:
            data_new[label] = feats
    
    return data_new
