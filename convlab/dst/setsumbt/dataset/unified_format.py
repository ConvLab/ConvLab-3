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
"""Convlab3 Unified Format Dialogue Datasets"""

from copy import deepcopy

import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm

from convlab.util import load_dataset
from convlab.dst.setsumbt.dataset.utils import (get_ontology_slots, ontology_add_values,
                                                get_values_from_data, ontology_add_requestable_slots,
                                                get_requestable_slots, load_dst_data, extract_dialogues,
                                                combine_value_sets, IdTensor)

transformers.logging.set_verbosity_error()


def convert_examples_to_features(data: list,
                                 ontology: dict,
                                 tokenizer: PreTrainedTokenizer,
                                 max_turns: int = 12,
                                 max_seq_len: int = 64) -> dict:
    """
    Convert dialogue examples to model input features and labels

    Args:
        data (list): List of all extracted dialogues
        ontology (dict): Ontology dictionary containing slots, slot descriptions and
        possible value sets including requests
        tokenizer (PreTrainedTokenizer): Tokenizer for the encoder model used
        max_turns (int): Maximum numbers of turns in a dialogue
        max_seq_len (int): Maximum number of tokens in a dialogue turn

    Returns:
        features (dict): All inputs and labels required to train the model
    """
    features = dict()
    ontology = deepcopy(ontology)

    # Get encoder input for system, user utterance pairs
    input_feats = []
    for dial in tqdm(data):
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
            # Truncate
            if len(dial_feats) >= max_turns:
                break
        input_feats.append(dial_feats)
    del dial_feats

    # Perform turn level padding
    dial_ids = list()
    for dial in data:
        _ids = [turn['dialogue_id'] for turn in dial][:max_turns]
        _ids += [''] * (max_turns - len(_ids))
        dial_ids.append(_ids)
    input_ids = [[turn['input_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                 for dial in input_feats]
    if 'token_type_ids' in input_feats[0][0]:
        token_type_ids = [[turn['token_type_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                          for dial in input_feats]
    else:
        token_type_ids = None
    if 'attention_mask' in input_feats[0][0]:
        attention_mask = [[turn['attention_mask'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                          for dial in input_feats]
    else:
        attention_mask = None
    del input_feats

    # Create torch data tensors
    features['dialogue_ids'] = IdTensor(dial_ids)
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
    for domslot in tqdm(informable_slots):
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                value = [v for d, substate in turn['state'].items() for s, v in substate.items()
                         if f'{d}-{s}' == domslot]
                domain, slot = domslot.split('-', 1)
                if turn['dataset_name'] in ontology[domain][slot]['dataset_names']:
                    value = value[0] if value else 'none'
                else:
                    value = -1
                if value in ontology[domain][slot]['possible_values'] and value != -1:
                    value = ontology[domain][slot]['possible_values'].index(value)
                else:
                    value = -1  # If value is not in ontology then we do not penalise the model
                labs.append(value)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['state_labels-' + domslot] = labels

    # Create requestable slot labels
    for domslot in tqdm(requestable_slots):
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                domain, slot = domslot.split('-', 1)
                if turn['dataset_name'] in ontology[domain][slot]['dataset_names']:
                    acts = [act['intent'] for act in turn['dialogue_acts']
                            if act['domain'] == domain and act['slot'] == slot]
                    if acts:
                        act_ = acts[0]
                        if act_ == 'request':
                            labs.append(1)
                        else:
                            labs.append(0)
                    else:
                        labs.append(0)
                else:
                    labs.append(-1)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['request_labels-' + domslot] = labels

    # General act labels (1-goodbye, 2-thank you)
    labels = []
    for dial in tqdm(data):
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
    features['general_act_labels'] = labels

    # Create active domain labels
    for domain in tqdm(domains):
        labels = []
        for dial in data:
            labs = []
            for turn in dial:
                possible_domains = list()
                for dom in ontology:
                    for slt in ontology[dom]:
                        if turn['dataset_name'] in ontology[dom][slt]['dataset_names']:
                            possible_domains.append(dom)

                if domain in turn['active_domains']:
                    labs.append(1)
                elif domain in possible_domains:
                    labs.append(0)
                else:
                    labs.append(-1)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['active_domain_labels-' + domain] = labels

    del labels

    return features


class UnifiedFormatDataset(Dataset):
    """
    Class for preprocessing, and storing data easily from the Convlab3 unified format.

    Attributes:
        dataset_dict (dict): Dictionary containing all the data in dataset
        ontology (dict): Set of all domain-slot-value triplets in the ontology of the model
        features (dict): Set of numeric features containing all inputs and labels formatted for the SetSUMBT model
    """
    def __init__(self,
                 dataset_name: str,
                 set_type: str,
                 tokenizer: PreTrainedTokenizer,
                 max_turns: int = 12,
                 max_seq_len: int = 64,
                 train_ratio: float = 1.0,
                 seed: int = 0,
                 data: dict = None,
                 ontology: dict = None):
        """
        Args:
            dataset_name (str): Name of the dataset/s to load (multiple to be seperated by +)
            set_type (str): Subset of the dataset to load (train, validation or test)
            tokenizer (transformers tokenizer): Tokenizer for the encoder model used
            max_turns (int): Maximum numbers of turns in a dialogue
            max_seq_len (int): Maximum number of tokens in a dialogue turn
            train_ratio (float): Fraction of training data to use during training
            seed (int): Seed governing random order of ids for subsampling
            data (dict): Dataset features for loading from dict
            ontology (dict): Ontology dict for loading from dict
        """
        if data is not None:
            self.ontology = ontology
            self.features = data
        else:
            if '+' in dataset_name:
                dataset_args = [{"dataset_name": name} for name in dataset_name.split('+')]
            else:
                dataset_args = [{"dataset_name": dataset_name}]
            self.dataset_dicts = [load_dataset(**dataset_args_) for dataset_args_ in dataset_args]
            self.ontology = get_ontology_slots(dataset_name)
            values = [get_values_from_data(dataset, set_type) for dataset in self.dataset_dicts]
            self.ontology = ontology_add_values(self.ontology, combine_value_sets(values), set_type)
            self.ontology = ontology_add_requestable_slots(self.ontology, get_requestable_slots(self.dataset_dicts))

            if train_ratio != 1.0:
                for dataset_args_ in dataset_args:
                    dataset_args_['dial_ids_order'] = seed
                    dataset_args_['split2ratio'] = {'train': train_ratio, 'validation': train_ratio}
            self.dataset_dicts = [load_dataset(**dataset_args_) for dataset_args_ in dataset_args]

            data = [load_dst_data(dataset_dict, data_split=set_type, speaker='all',
                                  dialogue_acts=True, split_to_turn=False)
                    for dataset_dict in self.dataset_dicts]
            data_list = [data_[set_type] for data_ in data]

            data = []
            for idx, data_ in enumerate(data_list):
                data += extract_dialogues(data_, dataset_args[idx]["dataset_name"])
            self.features = convert_examples_to_features(data, self.ontology, tokenizer, max_turns, max_seq_len)

    def __getitem__(self, index: int) -> dict:
        """
        Obtain dialogues with specific ids from dataset

        Args:
            index (int/list/tensor): Index/indices of dialogues to get

        Returns:
            features (dict): All inputs and labels required to train the model
        """
        return {label: self.features[label][index] for label in self.features
                if self.features[label] is not None}

    def __len__(self):
        """
        Get number of dialogues in the dataset

        Returns:
            len (int): Number of dialogues in the dataset object
        """
        return self.features['input_ids'].size(0)

    def resample(self, size: int = None) -> Dataset:
        """
        Resample subset of the dataset

        Args:
            size (int): Number of dialogues to sample

        Returns:
            self (Dataset): Dataset object
        """
        # If no subset size is specified we resample a set with the same size as the full dataset
        n_dialogues = self.__len__()
        if not size:
            size = n_dialogues

        dialogues = torch.randint(low=0, high=n_dialogues, size=(size,))
        self.features = self.__getitem__(dialogues)
        
        return self

    def to(self, device):
        """
        Map all data to a device

        Args:
            device (torch device): Device to map data to
        """
        self.device = device
        self.features = {label: self.features[label].to(device) for label in self.features
                         if self.features[label] is not None}

    @classmethod
    def from_datadict(cls, data: dict, ontology: dict):
        return cls(None, None, None, data=data, ontology=ontology)


def get_dataloader(dataset_name: str,
                   set_type: str,
                   batch_size: int,
                   tokenizer: PreTrainedTokenizer,
                   max_turns: int = 12,
                   max_seq_len: int = 64,
                   device='cpu',
                   resampled_size: int = None,
                   train_ratio: float = 1.0,
                   seed: int = 0) -> DataLoader:
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
        train_ratio (float): Ratio of training data to use for training
        seed (int): Seed governing random order of ids for subsampling

    Returns:
        loader (torch dataloader): Dataloader to train and evaluate the setsumbt model
    '''
    data = UnifiedFormatDataset(dataset_name, set_type, tokenizer, max_turns, max_seq_len, train_ratio=train_ratio,
                                seed=seed)
    data.to(device)

    if resampled_size:
        data = data.resample(resampled_size)

    if set_type in ['test', 'validation']:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return loader


def change_batch_size(loader: DataLoader, batch_size: int) -> DataLoader:
    """
    Change the batch size of a preloaded loader

    Args:
        loader (DataLoader): Dataloader to train and evaluate the setsumbt model
        batch_size (int): Batch size for the dataloader

    Returns:
        loader (DataLoader): Dataloader to train and evaluate the setsumbt model
    """

    if 'SequentialSampler' in str(loader.sampler):
        sampler = SequentialSampler(loader.dataset)
    else:
        sampler = RandomSampler(loader.dataset)
    loader = DataLoader(loader.dataset, sampler=sampler, batch_size=batch_size)

    return loader

def dataloader_sample_dialogues(loader: DataLoader, sample_size: int) -> DataLoader:
    """
    Sample a subset of the dialogues in a dataloader

    Args:
        loader (DataLoader): Dataloader to train and evaluate the setsumbt model
        sample_size (int): Number of dialogues to sample

    Returns:
        loader (DataLoader): Dataloader to train and evaluate the setsumbt model
    """

    dataset = loader.dataset.resample(sample_size)

    if 'SequentialSampler' in str(loader.sampler):
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    loader = DataLoader(loader.dataset, sampler=sampler, batch_size=loader.batch_size)

    return loader
