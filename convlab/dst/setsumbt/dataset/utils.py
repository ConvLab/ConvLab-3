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
"""Convlab3 Unified dataset data processing utilities"""

import numpy
import pdb

from convlab.util import load_ontology, load_dst_data, load_nlu_data
from convlab.dst.setsumbt.dataset.value_maps import VALUE_MAP, DOMAINS_MAP, QUANTITIES, TIME


def get_ontology_slots(dataset_name: str) -> dict:
    """
    Function to extract slots, slot descriptions and categorical slot values from the dataset ontology.

    Args:
        dataset_name (str): Dataset name

    Returns:
        ontology_slots (dict): Ontology dictionary containing slots, descriptions and categorical slot values
    """
    dataset_names = dataset_name.split('+') if '+' in dataset_name else [dataset_name]
    ontology_slots = dict()
    for dataset_name in dataset_names:
        ontology = load_ontology(dataset_name)
        domains = [domain for domain in ontology['domains'] if domain not in ['booking', 'general']]
        for domain in domains:
            domain_name = DOMAINS_MAP.get(domain, domain.lower())
            if domain_name not in ontology_slots:
                ontology_slots[domain_name] = dict()
            for slot, slot_info in ontology['domains'][domain]['slots'].items():
                if slot not in ontology_slots[domain_name]:
                    ontology_slots[domain_name][slot] = {'description': slot_info['description'],
                                                         'possible_values': list(),
                                                         'dataset_names': list()}
                if slot_info['is_categorical']:
                    ontology_slots[domain_name][slot]['possible_values'] += slot_info['possible_values']

                ontology_slots[domain_name][slot]['possible_values'] = list(set(ontology_slots[domain_name][slot]['possible_values']))
                ontology_slots[domain_name][slot]['dataset_names'].append(dataset_name)

    return ontology_slots


def get_values_from_data(dataset: dict, data_split: str = "train") -> dict:
    """
    Function to extract slots, slot descriptions and categorical slot values from the dataset ontology.

    Args:
        dataset (dict): Dataset dictionary obtained using the load_dataset function
        data_split (str): Dataset split: train/validation/test

    Returns:
        value_sets (dict): Dictionary containing possible values obtained from dataset
    """
    data = load_dst_data(dataset, data_split='all', speaker='user')

    # Remove test data from the data when building training/validation ontology
    if data_split == 'train':
        data = {key: itm for key, itm in data.items() if key == 'train'}
    elif data_split == 'validation':
        data = {key: itm for key, itm in data.items() if key in ['train', 'validation']}

    value_sets = {}
    for set_type, dataset in data.items():
        for turn in dataset:
            for domain, substate in turn['state'].items():
                domain_name = DOMAINS_MAP.get(domain, domain.lower())
                if domain_name not in value_sets:
                    value_sets[domain_name] = {}
                for slot, value in substate.items():
                    if slot not in value_sets[domain_name]:
                        value_sets[domain_name][slot] = []
                    if value and value not in value_sets[domain_name][slot]:
                        value_sets[domain_name][slot].append(value)
            # pdb.set_trace()

    return clean_values(value_sets)


def combine_value_sets(value_sets: list) -> dict:
    """
    Function to combine value sets extracted from different datasets

    Args:
        value_sets (list): List of value sets extracted using the get_values_from_data function

    Returns:
        value_set (dict): Dictionary containing possible values obtained from datasets
    """
    value_set = value_sets[0]
    for _value_set in value_sets[1:]:
        for domain, domain_info in _value_set.items():
            for slot, possible_values in domain_info.items():
                if domain not in value_set:
                    value_set[domain] = dict()
                if slot not in value_set[domain]:
                    value_set[domain][slot] = list()
                value_set[domain][slot] += _value_set[domain][slot]
                value_set[domain][slot] = list(set(value_set[domain][slot]))

    return value_set


def clean_values(value_sets: dict, value_map: dict = VALUE_MAP) -> dict:
    """
    Function to clean up the possible value sets extracted from the states in the dataset

    Args:
        value_sets (dict): Dictionary containing possible values obtained from dataset
        value_map (dict): Label map to avoid duplication and typos in values

    Returns:
        clean_vals (dict): Cleaned Dictionary containing possible values obtained from dataset
    """
    clean_vals = {}
    for domain, subset in value_sets.items():
        clean_vals[domain] = {}
        for slot, values in subset.items():
            # Remove pipe separated values
            values = list(set([val.split('|', 1)[0] for val in values]))

            # Map values using value_map
            for old, new in value_map.items():
                values = list(set([val.replace(old, new) for val in values]))

            # Remove empty and dontcare from possible value sets
            values = [val for val in values if val not in ['', 'dontcare']]

            # MultiWOZ specific value sets for quantity, time and boolean slots
            if 'people' in slot or 'duration' in slot or 'stay' in slot:
                values = QUANTITIES
            elif 'time' in slot or 'leave' in slot or 'arrive' in slot:
                values = TIME
            elif 'parking' in slot or 'internet' in slot:
                values = ['yes', 'no']

            clean_vals[domain][slot] = values

    return clean_vals


def ontology_add_values(ontology_slots: dict, value_sets: dict, data_split: str = "train") -> dict:
    """
    Add value sets obtained from the dataset to the ontology
    Args:
        ontology_slots (dict): Ontology dictionary containing slots, descriptions and categorical slot values
        value_sets (dict): Cleaned Dictionary containing possible values obtained from dataset
        data_split (str): Dataset split: train/validation/test

    Returns:
        ontology_slots (dict): Ontology dictionary containing slots, slot descriptions and possible value sets
    """
    ontology = {}
    for domain in sorted(ontology_slots):
        if data_split in ['train', 'validation']:
            if domain not in value_sets:
                continue
            possible_values = [v for slot, vals in value_sets[domain].items() for v in vals]
            if len(possible_values) == 0:
                continue
        ontology[domain] = {}
        for slot in sorted(ontology_slots[domain]):
            if not ontology_slots[domain][slot]['possible_values']:
                if domain in value_sets:
                    if slot in value_sets[domain]:
                        ontology_slots[domain][slot]['possible_values'] = value_sets[domain][slot]
            if ontology_slots[domain][slot]['possible_values']:
                values = sorted(ontology_slots[domain][slot]['possible_values'])
                ontology_slots[domain][slot]['possible_values'] = ['none', 'do not care'] + values

            ontology[domain][slot] = ontology_slots[domain][slot]

    return ontology


def get_requestable_slots(datasets: list) -> dict:
    """
    Function to get set of requestable slots from the dataset action labels.
    Args:
        datasets (dict): Dataset dictionary obtained using the load_dataset function

    Returns:
        slots (dict): Dictionary containing requestable domain-slot pairs
    """
    datasets = [load_nlu_data(dataset, data_split='all', speaker='user') for dataset in datasets]

    slots = {}
    for data in datasets:
        for set_type, subset in data.items():
            for turn in subset:
                requests = [act for act in turn['dialogue_acts']['categorical'] if act['intent'] == 'request']
                requests += [act for act in turn['dialogue_acts']['non-categorical'] if act['intent'] == 'request']
                requests += [act for act in turn['dialogue_acts']['binary'] if act['intent'] == 'request']
                requests = [(act['domain'], act['slot']) for act in requests]
                for domain, slot in requests:
                    domain_name = DOMAINS_MAP.get(domain, domain.lower())
                    if domain_name not in slots:
                        slots[domain_name] = []
                    slots[domain_name].append(slot)

    slots = {domain: list(set(slot_list)) for domain, slot_list in slots.items()}

    return slots


def ontology_add_requestable_slots(ontology_slots: dict, requestable_slots: dict) -> dict:
    """
    Add requestable slots obtained from the dataset to the ontology
    Args:
        ontology_slots (dict): Ontology dictionary containing slots, descriptions and categorical slot values
        requestable_slots (dict): Dictionary containing requestable domain-slot pairs

    Returns:
        ontology_slots (dict): Ontology dictionary containing slots, slot descriptions and
        possible value sets including requests
    """
    for domain in ontology_slots:
        for slot in ontology_slots[domain]:
            if domain in requestable_slots:
                if slot in requestable_slots[domain]:
                    ontology_slots[domain][slot]['possible_values'].append('?')

    return ontology_slots


def extract_turns(dialogue: list, dataset_name: str, dialogue_id: str) -> list:
    """
    Extract the required information from the data provided by unified loader
    Args:
        dialogue (list): List of turns within a dialogue
        dataset_name (str): Name of the dataset to which the dialogue belongs
        dialogue_str (str): ID of the dialogue

    Returns:
        turns (list): List of turns within a dialogue
    """
    turns = []
    turn_info = {}
    for turn in dialogue:
        if turn['speaker'] == 'system':
            turn_info['system_utterance'] = turn['utterance']

        # System utterance in the first turn is always empty as conversation is initiated by the user
        if turn['utt_idx'] == 1:
            turn_info['system_utterance'] = ''

        if turn['speaker'] == 'user':
            turn_info['user_utterance'] = turn['utterance']

            # Inform acts not required by model
            turn_info['dialogue_acts'] = [act for act in turn['dialogue_acts']['categorical']
                                          if act['intent'] not in ['inform']]
            turn_info['dialogue_acts'] += [act for act in turn['dialogue_acts']['non-categorical']
                                           if act['intent'] not in ['inform']]
            turn_info['dialogue_acts'] += [act for act in turn['dialogue_acts']['binary']
                                           if act['intent'] not in ['inform']]

            turn_info['state'] = turn['state']
            turn_info['dataset_name'] = dataset_name
            turn_info['dialogue_id'] = dialogue_id

        if 'system_utterance' in turn_info and 'user_utterance' in turn_info:
            turns.append(turn_info)
            turn_info = {}

    return turns


def clean_states(turns: list) -> list:
    """
    Clean the state within each turn of a dialogue (cleaning values and mapping to options used in ontology)
    Args:
        turns (list): List of turns within a dialogue

    Returns:
        clean_turns (list): List of turns within a dialogue
    """
    clean_turns = []
    for turn in turns:
        clean_state = {}
        clean_acts = []
        for act in turn['dialogue_acts']:
            domain = act['domain']
            act['domain'] = DOMAINS_MAP.get(domain, domain.lower())
            clean_acts.append(act)
        for domain, subset in turn['state'].items():
            domain_name = DOMAINS_MAP.get(domain, domain.lower())
            clean_state[domain_name] = {}
            for slot, value in subset.items():
                # Remove pipe separated values
                value = value.split('|', 1)[0]

                # Map values using value_map
                for old, new in VALUE_MAP.items():
                    value = value.replace(old, new)

                # Map dontcare to "do not care" and empty to 'none'
                value = value.replace('dontcare', 'do not care')
                value = value if value else 'none'

                # Map quantity values to the integer quantity value
                if 'people' in slot or 'duration' in slot or 'stay' in slot:
                    try:
                        if value not in ['do not care', 'none']:
                            value = int(value)
                            value = str(value) if value < 10 else QUANTITIES[-1]
                    except:
                        value = value
                # Map time values to the most appropriate value in the standard time set
                elif 'time' in slot or 'leave' in slot or 'arrive' in slot:
                    try:
                        if value not in ['do not care', 'none']:
                            # Strip after/before from time value
                            value = value.replace('after ', '').replace('before ', '')
                            # Extract hours and minutes from different possible formats
                            if ':' not in value and len(value) == 4:
                                h, m = value[:2], value[2:]
                            elif len(value) == 1:
                                h = int(value)
                                m = 0
                            elif 'pm' in value:
                                h = int(value.replace('pm', '')) + 12
                                m = 0
                            elif 'am' in value:
                                h = int(value.replace('pm', ''))
                                m = 0
                            elif ':' in value:
                                h, m = value.split(':')
                            elif ';' in value:
                                h, m = value.split(';')
                            # Map to closest 5 minutes
                            if int(m) % 5 != 0:
                                m = round(int(m) / 5) * 5
                                h = int(h)
                                if m == 60:
                                    m = 0
                                    h += 1
                                if h >= 24:
                                    h -= 24
                            # Set in standard 24 hour format
                            h, m = int(h), int(m)
                            value = '%02i:%02i' % (h, m)
                    except:
                        value = value
                # Map boolean slots to yes/no value
                elif 'parking' in slot or 'internet' in slot:
                    if value not in ['do not care', 'none']:
                        if value == 'free':
                            value = 'yes'
                        elif True in [v in value.lower() for v in ['yes', 'no']]:
                            value = [v for v in ['yes', 'no'] if v in value][0]

                clean_state[domain_name][slot] = value
        turn['state'] = clean_state
        turn['dialogue_acts'] = clean_acts
        clean_turns.append(turn)

    return clean_turns


def get_active_domains(turns: list) -> list:
    """
    Get active domains at each turn in a dialogue
    Args:
        turns (list): List of turns within a dialogue

    Returns:
        turns (list): List of turns within a dialogue
    """
    for turn_id in range(len(turns)):
        # At first turn all domains with not none values in the state are active
        if turn_id == 0:
            domains = [d for d, substate in turns[turn_id]['state'].items() for s, v in substate.items() if v != 'none']
            domains += [act['domain'] for act in turns[turn_id]['dialogue_acts'] if act['domain'] in turns[turn_id]['state']]
            domains = [DOMAINS_MAP.get(domain, domain.lower()) for domain in domains]
            turns[turn_id]['active_domains'] = list(set(domains))
        else:
            # Use changes in domains to identify active domains
            domains = []
            for domain, substate in turns[turn_id]['state'].items():
                domain_name = DOMAINS_MAP.get(domain, domain.lower())
                for slot, value in substate.items():
                    if value != turns[turn_id - 1]['state'][domain][slot]:
                        val = value
                    else:
                        val = 'none'
                    if value == 'none':
                        val = 'none'
                    if val != 'none':
                        domains.append(domain_name)
            # Add all domains activated by a user action
            domains += [act['domain'] for act in turns[turn_id]['dialogue_acts']
                        if act['domain'] in turns[turn_id]['state']]
            turns[turn_id]['active_domains'] = list(set(domains))

    return turns


class IdTensor:
    def __init__(self, values):
        self.values = numpy.array(values)

    def __getitem__(self, index: int):
        return self.values[index].tolist()

    def to(self, device):
        return self


def extract_dialogues(data: list, dataset_name: str) -> list:
    """
    Extract all dialogues from dataset
    Args:
        data (list): List of all dialogues in a subset of the data
        dataset_name (str): Name of the dataset to which the dialogues belongs

    Returns:
        dialogues (list): List of all extracted dialogues
    """
    dialogues = []
    for dial in data:
        dial_id = dial['dialogue_id']
        turns = extract_turns(dial['turns'], dataset_name, dial_id)
        turns = clean_states(turns)
        turns = get_active_domains(turns)
        dialogues.append(turns)

    return dialogues
