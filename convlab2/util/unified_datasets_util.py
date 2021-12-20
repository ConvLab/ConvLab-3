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


def load_dataset(dataset_name:str) -> Tuple[List, Dict]:
    """load unified datasets from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        dialogues (list): each element is a dialog in unified format
        ontology (dict): dataset ontology
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    with archive.open('data/ontology.json') as f:
        ontology = json.loads(f.read())
    return dialogues, ontology

def load_database(dataset_name:str):
    """load database from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        database: an instance of BaseDatabase
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    cwd = os.getcwd()
    os.chdir(data_dir)
    Database = importlib.import_module('database').Database
    os.chdir(cwd)
    database = Database()
    assert isinstance(database, BaseDatabase)
    return database

if __name__ == "__main__":
    dialogues, ontology = load_dataset('multiwoz21')
    database = load_database('multiwoz21')
    res = database.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arrive by', '11:15']], topk=3)
    print(res[0], len(res))
