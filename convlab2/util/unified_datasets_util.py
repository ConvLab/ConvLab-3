from typing import Dict, List, Tuple
from zipfile import ZipFile
import json
import os
import importlib
from abc import ABC, abstractmethod


class BaseDatabase(ABC):
    """Base class of unified database. Should override the query function."""
    def __init__(self):
        """extract data.zip and load the database."""

    @abstractmethod
    def query(self, domain:str, state:dict, topk:int, **kwargs)->list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""


def load_dataset(dataset_name:str) -> Tuple[Dict, Dict]:
    """load unified datasets from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        dataset (dict): keys are data splits and the values are lists of dialogues
        ontology (dict): dataset ontology
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/ontology.json') as f:
        ontology = json.loads(f.read())
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    dataset = {}
    for dialogue in dialogues:
        if dialogue['data_split'] not in dataset:
            dataset[dialogue['data_split']] = [dialogue]
        else:
            dataset[dialogue['data_split']].append(dialogue)
    return dataset, ontology

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

if __name__ == "__main__":
    # dataset, ontology = load_dataset('multiwoz21')
    # print(dataset.keys())
    # print(len(dataset['train']))    
    from convlab2.util.unified_datasets_util import BaseDatabase
    database = load_database('multiwoz21')
    res = database.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arrive by', '11:15']], topk=3)
    print(res[0], len(res))
