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
"""MultiWOZ 2.1/2.3 Dialogue Dataset"""

import os
import json
import requests
import zipfile
import io
from shutil import copy2 as copy

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from convlab2.dst.setsumbt.multiwoz.dataset.utils import (clean_text, ACTIVE_DOMAINS, get_domains, set_util_domains,
                                                        fix_delexicalisation, extract_dialogue, PRICERANGE,
                                                        BOOLEAN, DAYS, QUANTITIES, TIME, VALUE_MAP, map_values)


# Set up global data_directory
def set_datadir(dir):
    global DATA_DIR
    DATA_DIR = dir


def set_active_domains(domains):
    global ACTIVE_DOMAINS
    ACTIVE_DOMAINS = [d for d in domains if d in ACTIVE_DOMAINS]
    set_util_domains(ACTIVE_DOMAINS)


# MultiWOZ2.1 download link
URL = 'https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip'
def set_url(url):
    global URL
    URL = url


# Create Dialogue examples from the dataset
def create_examples(max_utt_len, get_requestable_slots=False, force_processing=False):

    # Load or download Raw Data
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'data_raw.json')):
        # Download data archive and extract
        archive = _download()
        data = _extract(archive)

        writer = open(os.path.join(DATA_DIR, 'data_raw.json'), 'w')
        json.dump(data, writer, indent = 2)
        del archive, writer
    else:
        reader = open(os.path.join(DATA_DIR, 'data_raw.json'), 'r')
        data = json.load(reader)

    if force_processing or not os.path.exists(os.path.join(DATA_DIR, 'data_train.json')):
        # Preprocess all dialogues
        data_processed = _process(data['data'], data['system_acts'])
        # Format data and split train, test and devlopment sets
        train, dev, test = _split_data(data_processed, data['testListFile'],
                                                            data['valListFile'], max_utt_len)

        # Write data
        writer = open(os.path.join(DATA_DIR, 'data_train.json'), 'w')
        json.dump(train, writer, indent = 2)
        writer = open(os.path.join(DATA_DIR, 'data_test.json'), 'w')
        json.dump(test, writer, indent = 2)
        writer = open(os.path.join(DATA_DIR, 'data_dev.json'), 'w')
        json.dump(dev, writer, indent = 2)
        writer.flush()
        writer.close()
        del writer

        # Extract slots and slot value candidates from the dataset
        for set_type in ['train', 'dev', 'test']:
            _get_ontology(set_type, get_requestable_slots)
        
        script_path = os.path.abspath(__file__).replace('/multiwoz21.py', '')
        file_name = 'mwoz21_ont_request.json' if get_requestable_slots else 'mwoz21_ont.json'
        copy(os.path.join(script_path, file_name), os.path.join(DATA_DIR, 'ontology_test.json'))
        copy(os.path.join(script_path, 'mwoz21_slot_descriptions.json'), os.path.join(DATA_DIR, 'slot_descriptions.json'))


# Extract slots and slot value candidates from the dataset
def _get_ontology(set_type, get_requestable_slots=False):

    datasets = ['train']
    if set_type in ['test', 'dev']:
        datasets.append('dev')
        datasets.append('test')

    # Load examples
    data = []
    for dataset in datasets:
        reader = open(os.path.join(DATA_DIR, 'data_%s.json' % dataset), 'r')
        data += json.load(reader)

    ontology = dict()
    for dial in data:
        for turn in dial['dialogue']:
            for state in turn['dialogue_state']:
                slot, value = state
                value = map_values(value)
                if slot not in ontology:
                    ontology[slot] = [value]
                else:
                    ontology[slot].append(value)

    requestable_slots = []
    if get_requestable_slots:
        for dial in data:
            for turn in dial['dialogue']:
                for act, dom, slot, val in turn['user_acts']:
                    if act == 'request':
                        requestable_slots.append(f'{dom}-{slot}')
        requestable_slots = list(set(requestable_slots))

    for slot in ontology:
        if 'price' in slot:
            ontology[slot] = PRICERANGE
        if 'parking' in slot or 'internet' in slot:
            ontology[slot] = BOOLEAN
        if 'day' in slot:
            ontology[slot] = DAYS
        if 'people' in slot or 'duration' in slot or 'stay' in slot:
            ontology[slot] = QUANTITIES
        if 'time' in slot or 'leave' in slot or 'arrive' in slot:
            ontology[slot] = TIME
        if 'stars' in slot:
            ontology[slot] += [str(i) for i in range(5)]

    # Sort slot values and add none and dontcare values
    for slot in ontology:
        ontology[slot] = list(set(ontology[slot]))
        ontology[slot] = ['none', 'do not care'] + sorted([s for s in ontology[slot] if s not in ['none', 'do not care']])
    for slot in requestable_slots:
        if slot in ontology:
            ontology[slot].append('request')
        else:
            ontology[slot] = ['request']

    writer = open(os.path.join(DATA_DIR, 'ontology_%s.json' % set_type), 'w')
    json.dump(ontology, writer, indent=2)
    writer.close()


# Convert dialogue examples to model input features and labels
def convert_examples_to_features(set_type, tokenizer, max_turns=12, max_seq_len=64):

    features = dict()

    # Load examples
    reader = open(os.path.join(DATA_DIR, 'data_%s.json' % set_type), 'r')
    data = json.load(reader)

    # Get encoder input for system, user utterance pairs
    input_feats = []
    for dial in data:
        dial_feats = []
        for turn in dial['dialogue']:
            if len(turn['system_transcript']) == 0:
                usr = turn['transcript']
                dial_feats.append(tokenizer.encode_plus(usr, add_special_tokens = True,
                                                        max_length = max_seq_len, padding='max_length',
                                                        truncation = 'longest_first'))
            else:
                usr = turn['transcript']
                sys = turn['system_transcript']
                dial_feats.append(tokenizer.encode_plus(usr, sys, add_special_tokens = True,
                                                        max_length = max_seq_len, padding='max_length',
                                                        truncation = 'longest_first'))
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

    # Load ontology
    reader = open(os.path.join(DATA_DIR, 'ontology_%s.json' % set_type), 'r')
    ontology = json.load(reader)
    reader.close()

    informable_slots = [slot for slot, values in ontology.items() if values != ['request']]
    requestable_slots = [slot for slot, values in ontology.items() if 'request' in values]
    for slot in requestable_slots:
        ontology[slot].remove('request')
    
    domains = list(set(informable_slots + requestable_slots))
    domains = list(set([slot.split('-', 1)[0] for slot in domains]))

    # Create slot labels
    for slot in informable_slots:
        labels = []
        for dial in data:
            labs = []
            for turn in dial['dialogue']:
                slots_active = [s for s, v in turn['dialogue_state']]
                if slot in slots_active:
                    value = [v for s, v in turn['dialogue_state'] if s == slot][0]
                else:
                    value = 'none'
                if value in ontology[slot]:
                    value = ontology[slot].index(value)
                else:
                    value = map_values(value)
                    if value in ontology[slot]:
                        value = ontology[slot].index(value)
                    else:
                        value = -1 # If value is not in ontology then we do not penalise the model
                labs.append(value)
                if len(labs) >= max_turns:
                    break
            labs = labs + [-1] * (max_turns - len(labs))
            labels.append(labs)

        labels = torch.tensor(labels)
        features['labels-' + slot] = labels

    for slot in requestable_slots:
        labels = []
        for dial in data:
            labs = []
            for turn in dial['dialogue']:
                slots_active = [[d, s] for i, d, s, v in turn['user_acts']]
                if slot.split('-', 1) in slots_active:
                    act_ = [i for i, d, s, v in turn['user_acts'] if f"{d}-{s}" == slot][0]
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
        features['request-' + slot] = labels
    
    # Greeting act labels (0-no greeting, 1-goodbye, 2-thank you)
    labels = []
    for dial in data:
        labs = []
        for turn in dial['dialogue']:
            greeting_active = [i for i, d, s, v in turn['user_acts'] if i in ['bye', 'thank']]
            if greeting_active:
                if 'bye' in greeting_active:
                    labs.append(1)
                else :
                    labs.append(2)
            else:
                labs.append(0)
            if len(labs) >= max_turns:
                break
        labs = labs + [-1] * (max_turns - len(labs))
        labels.append(labs)
    
    labels = torch.tensor(labels)
    features['goodbye'] = labels

    for domain in domains:
        labels = []
        for dial in data:
            labs = []
            for turn in dial['dialogue']:
                if domain == turn['domain']:
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


# MultiWOZ2.1 Dataset object
class MultiWoz21(Dataset):

    def __init__(self, set_type, tokenizer, max_turns=12, max_seq_len=64):
        self.features = convert_examples_to_features(set_type, tokenizer, max_turns, max_seq_len)

    def __getitem__(self, index):
        return {label: self.features[label][index] for label in self.features
                if self.features[label] is not None}

    def __len__(self):
        return self.features['input_ids'].size(0)

    def resample(self, size=None):
        n_dialogues = self.__len__()
        if not size:
            size = n_dialogues

        dialogues = torch.randint(low=0, high=n_dialogues, size=(size,))
        self.features = {label: self.features[label][dialogues] for label in self.features
                        if self.features[label] is not None}
        
        return self

    def to(self, device):
        self.device = device
        self.features = {label: self.features[label].to(device) for label in self.features
                         if self.features[label] is not None}


# MultiWOZ2.1 Dataset object
class EnsembleMultiWoz21(Dataset):
    def __init__(self, data):
        self.features = data

    def __getitem__(self, index):
        return {label: self.features[label][index] for label in self.features
                if self.features[label] is not None}

    def __len__(self):
        return self.features['input_ids'].size(0)

    def resample(self, size=None):
        n_dialogues = self.__len__()
        if not size:
            size = n_dialogues

        dialogues = torch.randint(low=0, high=n_dialogues, size=(size,))
        self.features = {label: self.features[label][dialogues] for label in self.features
                        if self.features[label] is not None}

    def to(self, device):
        self.device = device
        self.features = {label: self.features[label].to(device) for label in self.features
                         if self.features[label] is not None}


# Module to create torch dataloaders
def get_dataloader(set_type, batch_size, tokenizer, max_turns=12, max_seq_len=64, device=None, resampled_size=None):
    data = MultiWoz21(set_type, tokenizer, max_turns, max_seq_len)
    data.to('cpu')

    if resampled_size:
        data.resample(resampled_size)

    if set_type in ['test', 'dev']:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return loader


def _download(chunk_size=1048576):
    """Download data archive.

    Parameters:
        chunk_size (int): Download chunk size. (default=1048576)
    Returns:
        archive: ZipFile archive object.
    """
    # Download the archive byte string
    req = requests.get(URL, stream=True)
    archive = b''
    for n_chunks, chunk in tqdm(enumerate(req.iter_content(chunk_size=chunk_size)), desc='Download Chunk'):
        if chunk:
            archive += chunk

    # Convert the bytestring into a zipfile object
    archive = io.BytesIO(archive)
    archive = zipfile.ZipFile(archive)

    return archive


def _extract(archive):
    """Extract the json dictionaries from the archive.

    Parameters:
        archive: ZipFile archive object.
    Returns:
        data: Data dictionary.
    """
    files = [file for file in archive.filelist if ('.json' in file.filename or '.txt' in file.filename)
            and 'MACOSX' not in file.filename]
    objects = []
    for file in tqdm(files, desc='File'):
        data = archive.open(file).read()
        # Get data objects from the files
        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError:
            data = data.decode().split('\n')
        objects.append(data)

    files = [file.filename.split('/')[-1].split('.')[0] for file in files]

    data = {file: data for file, data in zip(files, objects)}
    return data


# Process files
def _process(dialogue_data, acts_data):
    print('Processing Dialogues')
    out = {}
    for dial_name in tqdm(dialogue_data):
        dialogue = dialogue_data[dial_name]

        prev_dom = ''
        for turn_id, turn in enumerate(dialogue['log']):
            dialogue['log'][turn_id]['text'] = clean_text(turn['text'])
            if len(turn['metadata']) != 0:
                crnt_dom = get_domains(dialogue['log'], turn_id, prev_dom)
                prev_dom = crnt_dom
                dialogue['log'][turn_id - 1]['domain'] = crnt_dom

            dialogue['log'][turn_id] = fix_delexicalisation(turn)

        out[dial_name] = dialogue

    return out


# Split data (train, dev, test)
def _split_data(dial_data, test, dev, max_utt_len):
    train_dials, test_dials, dev_dials = [], [], []
    print('Formatting and Splitting Data')
    for name in tqdm(dial_data):
        dialogue = dial_data[name]
        domains = []

        dial = extract_dialogue(dialogue, max_utt_len)
        if dial:
            dialogue = dict()
            dialogue['dialogue_idx'] = name
            dialogue['domains'] = []
            dialogue['dialogue'] = []

            for turn_id, turn in enumerate(dial):
                turn_dialog = dict()
                turn_dialog['system_transcript'] = dial[turn_id - 1]['sys'] if turn_id > 0 else ''
                turn_dialog['turn_idx'] = turn_id
                turn_dialog['dialogue_state'] = turn['ds']
                turn_dialog['transcript'] = turn['usr']
                # turn_dialog['system_acts'] = dial[turn_id - 1]['sys_a'] if turn_id > 0 else []
                turn_dialog['user_acts'] = turn['usr_a']
                turn_dialog['domain'] = turn['domain']
                dialogue['domains'].append(turn['domain'])
                dialogue['dialogue'].append(turn_dialog)

            dialogue['domains'] = [d for d in list(set(dialogue['domains'])) if d != '']
            if True in [dom not in ACTIVE_DOMAINS for dom in dialogue['domains']]:
                dialogue['domains'] = []
            dialogue['domains'] = [dom for dom in dialogue['domains'] if dom in ACTIVE_DOMAINS]

            if dialogue['domains']:
                if name in test:
                    test_dials.append(dialogue)
                elif name in dev:
                    dev_dials.append(dialogue)
                else:
                    train_dials.append(dialogue)

    print('Number of Dialogues:\nTrain: %i\nDev: %i\nTest: %i' % (len(train_dials), len(dev_dials), len(test_dials)))

    return train_dials, dev_dials, test_dials
