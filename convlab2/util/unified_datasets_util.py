from copy import deepcopy
from typing import Dict, List, Tuple
from zipfile import ZipFile
import json
import os
import re
import importlib
from abc import ABC, abstractmethod
from pprint import pprint


class BaseDatabase(ABC):
    """Base class of unified database. Should override the query function."""
    def __init__(self):
        """extract data.zip and load the database."""

    @abstractmethod
    def query(self, domain:str, state:dict, topk:int, **kwargs)->list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""


def load_dataset(dataset_name:str, dial_ids_order=None) -> Dict:
    """load unified dataset from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`
        dial_ids_order (int): idx of shuffled dial order in `data/unified_datasets/$dataset_name/shuffled_dial_ids.json`

    Returns:
        dataset (dict): keys are data splits and the values are lists of dialogues
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    dataset = {}
    if dial_ids_order is not None:
        dial_ids = json.load(open(os.path.join(data_dir, 'shuffled_dial_ids.json')))[dial_ids_order]
        for data_split in dial_ids:
            dataset[data_split] = [dialogues[i] for i in dial_ids[data_split]]
    else:
        for dialogue in dialogues:
            if dialogue['data_split'] not in dataset:
                dataset[dialogue['data_split']] = [dialogue]
            else:
                dataset[dialogue['data_split']].append(dialogue)
    return dataset

def load_ontology(dataset_name:str) -> Dict:
    """load unified ontology from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        ontology (dict): dataset ontology
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/ontology.json') as f:
        ontology = json.loads(f.read())
    return ontology

def load_database(dataset_name:str):
    """load database from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        database: an instance of BaseDatabase
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}/database.py'))
    module_spec = importlib.util.spec_from_file_location('database', data_dir)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    Database = module.Database
    assert issubclass(Database, BaseDatabase)
    database = Database()
    assert isinstance(database, BaseDatabase)
    return database

def load_unified_data(
    dataset, 
    data_split='all', 
    speaker='all', 
    utterance=False, 
    dialogue_acts=False, 
    state=False, 
    db_results=False,
    use_context=False, 
    context_window_size=0, 
    terminated=False, 
    goal=False, 
    active_domains=False,
    split_to_turn=True
):
    data_splits = dataset.keys() if data_split == 'all' else [data_split]
    assert speaker in ['user', 'system', 'all']
    assert not use_context or context_window_size > 0
    info_list = list(filter(eval, ['utterance', 'dialogue_acts', 'state', 'db_results']))
    info_list += ['utt_idx']
    data_by_split = {}
    for data_split in data_splits:
        data_by_split[data_split] = []
        for dialogue in dataset[data_split]:
            context = []
            for turn in dialogue['turns']:
                sample = {'speaker': turn['speaker']}
                for ele in info_list:
                    if ele in turn:
                        sample[ele] = turn[ele]
                
                if use_context or not split_to_turn:
                    sample_copy = deepcopy(sample)
                    context.append(sample_copy)

                if split_to_turn and speaker in [turn['speaker'], 'all']:
                    if use_context:
                        sample['context'] = context[-context_window_size-1:-1]
                    if goal:
                        sample['goal'] = dialogue['goal']
                    if active_domains:
                        sample['domains'] = dialogue['domains']
                    if terminated:
                        sample['terminated'] = turn['utt_idx'] == len(dialogue['turns']) - 1
                    if speaker == 'system' and 'booked' in turn:
                        sample['booked'] = turn['booked']
                    data_by_split[data_split].append(sample)
            if not split_to_turn:
                dialogue['turns'] = context
                data_by_split[data_split].append(dialogue)
    return data_by_split


def load_nlu_data(dataset, data_split='all', speaker='user', use_context=False, context_window_size=0, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', use_context)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('dialogue_acts', True)
    return load_unified_data(dataset, **kwargs)


def load_dst_data(dataset, data_split='all', speaker='user', context_window_size=100, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('state', True)
    return load_unified_data(dataset, **kwargs)

def load_policy_data(dataset, data_split='all', speaker='system', context_window_size=1, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('state', True)
    kwargs.setdefault('db_results', True)
    kwargs.setdefault('dialogue_acts', True)
    kwargs.setdefault('terminated', True)
    return load_unified_data(dataset, **kwargs)

def load_nlg_data(dataset, data_split='all', speaker='system', use_context=False, context_window_size=0, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', use_context)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('dialogue_acts', True)
    return load_unified_data(dataset, **kwargs)

def load_e2e_data(dataset, data_split='all', speaker='system', context_window_size=100, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('state', True)
    kwargs.setdefault('db_results', True)
    kwargs.setdefault('dialogue_acts', True)
    return load_unified_data(dataset, **kwargs)

def load_rg_data(dataset, data_split='all', speaker='system', context_window_size=100, **kwargs):
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    return load_unified_data(dataset, **kwargs)


def create_delex_data(dataset, delex_func=lambda d,s,v: f'[({d})-({s})]', ignore_values=['yes', 'no']):
    """add delex_utterance to the dataset according to dialogue acts and belief_state
    delex_func: function that return the placeholder (e.g. "[(domain_name)-(slot_name)]") given (domain, slot, value)
    ignore_values: ignored values when delexicalizing using the categorical acts and states
    """
    # 

    def delex_inplace(texts_placeholders, value_pattern):
        res = []
        for substring, is_placeholder in texts_placeholders:
            if not is_placeholder:
                matches = value_pattern.findall(substring)
                res.append(len(matches) == 1)
            else:
                res.append(False)
        if sum(res) == 1:
            # only one piece matches
            idx = res.index(True)
            substring = texts_placeholders[idx][0]
            searchObj = re.search(value_pattern, substring)
            assert searchObj
            start, end = searchObj.span(1)
            texts_placeholders[idx:idx+1] = [(substring[0:start], False), (placeholder, True), (substring[end:], False)]
            return True
        return False

    delex_vocab = set()
    for data_split in dataset:
        for dialog in dataset[data_split]:
            state = {}
            for turn in dialog['turns']:
                utt = turn['utterance']
                delex_utt = []
                last_end = 0
                # ignore the non-categorical das that do not have span annotation
                spans = [x for x in turn['dialogue_acts']['non-categorical'] if 'start' in x]
                for da in sorted(spans, key=lambda x: x['start']):
                    # from left to right
                    start, end = da['start'], da['end']
                    domain, slot, value = da['domain'], da['slot'], da['value']
                    assert utt[start:end] == value
                    # make sure there are no words/number prepend & append and no overlap with other spans
                    if start >= last_end and (start == 0 or re.match('\W', utt[start-1])) and (end == len(utt) or re.match('\W', utt[end])):
                        placeholder = delex_func(domain, slot, value)
                        delex_vocab.add(placeholder)
                        delex_utt.append((utt[last_end:start], False))
                        delex_utt.append((placeholder, True))
                        last_end = end
                delex_utt.append((utt[last_end:], False))

                # search for value in categorical dialogue acts and belief state
                for da in sorted(turn['dialogue_acts']['categorical'], key=lambda x: len(x['value'])):
                    domain, slot, value = da['domain'], da['slot'], da['value']
                    if value.lower() not in ignore_values:
                        placeholder = delex_func(domain, slot, value)
                        pattern = re.compile(r'\b({})\b'.format(value), flags=re.I)
                        if delex_inplace(delex_utt, pattern):
                            delex_vocab.add(placeholder)

                # for domain in turn['state']
                if 'state' in turn:
                    state = turn['state']
                for domain in state:
                    for slot, values in state[domain].items():
                        if len(values) > 0:
                            # has value
                            for value in values.split('|'):
                                if value.lower() not in ignore_values:
                                    placeholder = delex_func(domain, slot, value)
                                    pattern = re.compile(r'\b({})\b'.format(value), flags=re.I)
                                    if delex_inplace(delex_utt, pattern):
                                        delex_vocab.add(placeholder)

                turn['delex_utterance'] = ''.join([x[0] for x in delex_utt])
    
    return dataset, sorted(list(delex_vocab))


if __name__ == "__main__":
    dataset = load_dataset('multiwoz21', dial_ids_order=0)
    train_ratio = 0.1
    dataset['train'] = dataset['train'][:round(len(dataset['train'])*train_ratio)]
    print(len(dataset['train']))
    print(dataset.keys())
    print(len(dataset['test']))

    from convlab2.util.unified_datasets_util import BaseDatabase
    database = load_database('multiwoz21')
    res = database.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arrive by', '11:15']], topk=3)
    print(res[0], len(res))
    
    data_by_split = load_nlu_data(dataset, data_split='test', speaker='user')
    pprint(data_by_split['test'][0])

    def delex_slot(domain, slot, value):
        # only use slot name for delexicalization
        return f'[{slot}]'

    dataset, delex_vocab = create_delex_data(dataset, delex_slot)
    json.dump(dataset['test'], open('new_delex_multiwoz21_test.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(delex_vocab, open('new_delex_vocab.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with open('new_delex_cmp.txt', 'w') as f:
        for dialog in dataset['test']:
            for turn in dialog['turns']:
                f.write(turn['utterance']+'\n')
                f.write(turn['delex_utterance']+'\n')
                f.write('\n')
