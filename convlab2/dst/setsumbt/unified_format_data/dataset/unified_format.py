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
"""Convlab3 Unified Format Dialogue Dataset"""

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from copy import deepcopy

from convlab2.dst.setsumbt.unified_format_data.dataset.utils import (load_dataset, get_ontology_slots,
                                            ontology_add_values, get_values_from_data, ontology_add_requestable_slots,
                                            get_requestable_slots, load_dst_data, extract_dialogues)


# Convert dialogue examples to model input features and labels
def convert_examples_to_features(data: list, ontology: dict, tokenizer, max_turns: int=12, max_seq_len: int=64) -> dict:
    '''
    Convert dialogue examples to model input features and labels
    Args:
        data (list): List of all extracted dialogues
        ontology (dict): Ontology dictionary containing slots, slot descriptions and
        possible value sets including requests
        tokenizer (transformers tokenizer): Tokenizer for the encoder model used
        max_turns (int): Maximum numbers of turns in a dialogue
        max_seq_len (int): Maximum number of tokens in a dialogue turn

    Returns:
        features (dict): All inputs and labels required to train the model
    '''
    features = dict()
    ontology = deepcopy(ontology)

    # Get encoder input for system, user utterance pairs
    input_feats = []
    for dial in data:
        dial_feats = []
        for turn in dial:
            if len(turn['system_utterance']) == 0:
                usr = turn['user_utterance']
                dial_feats.append(tokenizer.encode_plus(usr, add_special_tokens=True,
                                                        max_length=max_seq_len, padding='max_length',
                                                        truncation='longest_first'))
            else:
                usr = turn['user_utterance']
                sys = turn['system_utterance']
                dial_feats.append(tokenizer.encode_plus(usr, sys, add_special_tokens=True,
                                                        max_length=max_seq_len, padding='max_length',
                                                        truncation='longest_first'))
            # Trucate
            if len(dial_feats) >= max_turns:
                break
        input_feats.append(dial_feats)
    del dial_feats

    # Perform turn level padding
    input_ids = [[turn['input_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial)) for dial in input_feats]
    if 'token_type_ids' in input_feats[0][0]:
        token_type_ids = [[turn['token_type_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial)) for dial in input_feats]
    else:
        token_type_ids = None
    if 'attention_mask' in input_feats[0][0]:
        attention_mask = [[turn['attention_mask'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial)) for dial in input_feats]
    else:
        attention_mask = None
    del input_feats

    # Create torch data tensors
    features['input_ids'] = torch.tensor(input_ids)
    features['token_type_ids'] = torch.tensor(token_type_ids) if token_type_ids else None
    features['attention_mask'] = torch.tensor(attention_mask) if attention_mask else None
    del input_ids, token_type_ids, attention_mask

    # Extract all informable and requestable slots from the ontology
    informable_slots = [f"{domain}-{slot}" for domain in ontology for slot in ontology[domain]
                        if ontology[domain][slot]['possible_values']
                        and ontology[domain][slot]['possible_values'] != ['?']]
    requestable_slots = [f"{domain}-{slot}" for domain in ontology for slot in ontology[domain]
                         if '?' in ontology[domain][slot]['possible_values']]
    for slot in requestable_slots:
        domain, slot = slot.split('-', 1)
        ontology[domain][slot]['possible_values'].remove('?')

    # Extract a list of domains from the ontology slots
    domains = list(set(informable_slots + requestable_slots))
    domains = list(set([slot.split('-', 1)[0] for slot in domains]))

    # Create slot labels
    for domslot in informable_slots:
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                value = [v for d, substate in turn['state'].items() for s, v in substate.items() if f'{d}-{s}' == domslot][0]
                domain, slot = domslot.split('-', 1)
                if value in ontology[domain][slot]['possible_values']:
                    value = ontology[domain][slot]['possible_values'].index(value)
                else:
                    value = -1 # If value is not in ontology then we do not penalise the model
                labs.append(value)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['labels-' + domslot] = labels

    # Create requestable slot labels
    for domslot in requestable_slots:
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                domain, slot = domslot.split('-', 1)
                acts = [act['intent'] for act in turn['dialogue_acts'] if act['domain'] == domain and act['slot'] == slot]
                if acts:
                    act_ = acts[0]
                    if act_ == 'request':
                        labs.append(1)
                    else:
                        labs.append(0)
                else:
                    labs.append(0)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['request-' + domslot] = labels

    # Greeting act labels (0-no greeting, 1-goodbye, 2-thank you)
    labels = []
    for dial in data:
        labs = []
        for turn in dial:
            acts = [act['intent'] for act in turn['dialogue_acts'] if act['intent'] in ['bye', 'thank']]
            if acts:
                if 'bye' in acts:
                    labs.append(1)
                else:
                    labs.append(2)
            else:
                labs.append(0)
            if len(labs) >= max_turns:
                break
        labs = labs + [-1] * (max_turns - len(labs))
        labels.append(labs)

    labels = torch.tensor(labels)
    features['goodbye'] = labels

    # Create active domain labels
    for domain in domains:
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                if domain in turn['active_domains']:
                    labs.append(1)
                else:
                    labs.append(0)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['active-' + domain] = labels

    del labels

    return features


# Unified Dataset object
class UnifiedDataset(Dataset):

    def __init__(self, dataset_name: str, set_type: str, tokenizer, max_turns: int=12, max_seq_len:int =64):
        '''
        Build Unified Dataset object
        Args:
            dataset_name (str): Name of the dataset to load
            set_type (str): Subset of the dataset to load (train, validation or test)
            tokenizer (transformers tokenizer): Tokenizer for the encoder model used
            max_turns (int): Maximum numbers of turns in a dialogue
            max_seq_len (int): Maximum number of tokens in a dialogue turn
        '''
        self.dataset_dict = load_dataset(dataset_name)
        self.ontology = get_ontology_slots(dataset_name)
        self.ontology = ontology_add_values(self.ontology, get_values_from_data(self.dataset_dict))
        self.ontology = ontology_add_requestable_slots(self.ontology, get_requestable_slots(self.dataset_dict))

        data = load_dst_data(self.dataset_dict, data_split=set_type, speaker='all', dialogue_acts=True, split_to_turn=False)
        data = extract_dialogues(data[set_type])
        self.features = convert_examples_to_features(data, self.ontology, tokenizer, max_turns, max_seq_len)

    def __getitem__(self, index):
        return {label: self.features[label][index] for label in self.features
                if self.features[label] is not None}

    def __len__(self):
        return self.features['input_ids'].size(0)

    # Resample subset of the dataset
    def resample(self, size=None):
        '''
        Resample subset of the dataset
        Args:
            size (int): Number of dialogues to sample
        '''
        # If no subset size is specified we resample a set with the same size as the full dataset
        n_dialogues = self.__len__()
        if not size:
            size = n_dialogues

        dialogues = torch.randint(low=0, high=n_dialogues, size=(size,))
        self.features = {label: self.features[label][dialogues] for label in self.features
                        if self.features[label] is not None}
        
        return self

    # Map all data to a device
    def to(self, device):
        '''
        Map all data to a device
        Args:
            device (torch device): Device to map data to
        '''
        self.device = device
        self.features = {label: self.features[label].to(device) for label in self.features
                         if self.features[label] is not None}


# Module to create torch dataloaders
def get_dataloader(dataset_name: str, set_type: str, batch_size: int, tokenizer, max_turns: int=12, max_seq_len: int=64,
                   device='cpu', resampled_size=None):
    '''
    Module to create torch dataloaders
    Args:
        dataset_name (str): Name of the dataset to load
        set_type (str): Subset of the dataset to load (train, validation or test)
        batch_size (int): Batch size for the dataloader
        tokenizer (transformers tokenizer): Tokenizer for the encoder model used
        max_turns (int): Maximum numbers of turns in a dialogue
        max_seq_len (int): Maximum number of tokens in a dialogue turn
        device (torch device): Device to map data to
        resampled_size (int): Number of dialogues to sample

    Returns:
        loader (torch dataloader): Dataloader to train and evaluate the setsumbt model
    '''
    data = UnifiedDataset(dataset_name, set_type, tokenizer, max_turns, max_seq_len)
    data.to(device)

    if resampled_size:
        data = data.resample(resampled_size)

    if set_type in ['test', 'validation']:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return loader
