from copy import deepcopy
from typing import Dict, List, Tuple
from zipfile import ZipFile
import json
import os
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


def load_dataset(dataset_name:str) -> Dict:
    """load unified dataset from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        dataset (dict): keys are data splits and the values are lists of dialogues
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    dataset = {}
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
                    if speaker == 'system':
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

if __name__ == "__main__":
    dataset = load_dataset('multiwoz21')
    print(dataset.keys())
    print(len(dataset['test']))

    from convlab2.util.unified_datasets_util import BaseDatabase
    database = load_database('multiwoz21')
    res = database.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arrive by', '11:15']], topk=3)
    print(res[0], len(res))
    
    data_by_split = load_nlu_data(dataset, data_split='test', speaker='user')
    pprint(data_by_split['test'][0])
