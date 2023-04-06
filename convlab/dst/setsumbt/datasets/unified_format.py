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
"""Convlab3 Unified Format Dialogue Datasets"""

import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers.tokenization_utils import PreTrainedTokenizer

from convlab.util import load_dataset
from convlab.dst.setsumbt.datasets.utils import (get_ontology_slots, ontology_add_values,
                                                 get_values_from_data, ontology_add_requestable_slots,
                                                 get_requestable_slots, load_dst_data, extract_dialogues,
                                                 combine_value_sets)

transformers.logging.set_verbosity_error()


class UnifiedFormatDataset(Dataset):
    """
    Class for preprocessing, and storing data easily from the Convlab3 unified format.

    Attributes:
        set_type (str): Subset of the dataset to load (train, validation or test)
        dataset_dicts (dict): Dictionary containing all the data in dataset
        ontology (dict): Set of all domain-slot-value triplets in the ontology of the model
        ontology_embeddings (dict): Set of all domain-slot-value triplets in the ontology of the model
        features (dict): Set of numeric features containing all inputs and labels formatted for the SetSUMBT model
    """
    def __init__(self,
                 dataset_name: str,
                 set_type: str,
                 tokenizer: PreTrainedTokenizer,
                 ontology_encoder,
                 max_turns: int = 12,
                 max_seq_len: int = 64,
                 train_ratio: float = 1.0,
                 seed: int = 0,
                 data: dict = None,
                 ontology: dict = None,
                 ontology_embeddings: dict = None):
        """
        Args:
            dataset_name (str): Name of the dataset/s to load (multiple to be seperated by +)
            set_type (str): Subset of the dataset to load (train, validation or test)
            tokenizer (transformers tokenizer): Tokenizer for the encoder model used
            ontology_encoder (transformers model): Ontology encoder model
            max_turns (int): Maximum numbers of turns in a dialogue
            max_seq_len (int): Maximum number of tokens in a dialogue turn
            train_ratio (float): Fraction of training data to use during training
            seed (int): Seed governing random order of ids for subsampling
            data (dict): Dataset features for loading from dict
            ontology (dict): Ontology dict for loading from dict
            ontology_embeddings (dict): Ontology embeddings for loading from dict
        """
        # Load data from dict if provided
        if data is not None:
            self.set_type = set_type
            self.ontology = ontology
            self.ontology_embeddings = ontology_embeddings
            self.features = data
        # Load data from dataset if data is not provided
        else:
            if '+' in dataset_name:
                dataset_args = [{"dataset_name": name} for name in dataset_name.split('+')]
            else:
                dataset_args = [{"dataset_name": dataset_name}]
            self.dataset_dicts = [load_dataset(**dataset_args_) for dataset_args_ in dataset_args]
            self.set_type = set_type

            self.ontology = get_ontology_slots(dataset_name)
            values = [get_values_from_data(dataset, set_type) for dataset in self.dataset_dicts]
            self.ontology = ontology_add_values(self.ontology, combine_value_sets(values), set_type)
            self.ontology = ontology_add_requestable_slots(self.ontology, get_requestable_slots(self.dataset_dicts))

            tokenizer.set_setsumbt_ontology(self.ontology)
            self.ontology_embeddings = ontology_encoder.get_slot_candidate_embeddings()

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
            self.features = tokenizer.encode(data, max_turns, max_seq_len)

    def __getitem__(self, index: int) -> dict:
        """
        Obtain dialogues with specific ids from dataset

        Args:
            index (int/list/tensor): Index/indices of dialogues to get

        Returns:
            features (dict): All inputs and labels required to train the model
        """
        feats = dict()
        for label in self.features:
            if self.features[label] is not None:
                if label == 'dialogue_ids':
                    if type(index) == int:
                        feat = self.features[label][index]
                    else:
                        feat = [self.features[label][idx] for idx in index]
                else:
                    feat = self.features[label][index]

                feats[label] = feat

        return feats

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
    def from_datadict(cls, set_type: str, data: dict, ontology: dict, ontology_embeddings: dict):
        return cls(None, set_type, None, None, data=data, ontology=ontology, ontology_embeddings=ontology_embeddings)


def get_dataloader(dataset_name: str,
                   set_type: str,
                   batch_size: int,
                   tokenizer: PreTrainedTokenizer,
                   ontology_encoder,
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
        ontology_encoder (OntologyEncoder): Ontology encoder object
        max_turns (int): Maximum numbers of turns in a dialogue
        max_seq_len (int): Maximum number of tokens in a dialogue turn
        device (torch device): Device to map data to
        resampled_size (int): Number of dialogues to sample
        train_ratio (float): Ratio of training data to use for training
        seed (int): Seed governing random order of ids for subsampling

    Returns:
        loader (torch dataloader): Dataloader to train and evaluate the setsumbt model
    '''
    data = UnifiedFormatDataset(dataset_name, set_type, tokenizer, ontology_encoder, max_turns, max_seq_len,
                                train_ratio=train_ratio, seed=seed)
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
