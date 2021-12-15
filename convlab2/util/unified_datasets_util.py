from zipfile import ZipFile
import json
import os
import importlib

def load_dataset(dataset_name):
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    archive = ZipFile(os.path.join(data_dir, 'data.zip'))
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    with archive.open('data/ontology.json') as f:
        ontology = json.loads(f.read())
    return dialogues, ontology

def load_database(dataset_name):
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../../data/unified_datasets/{dataset_name}'))
    cwd = os.getcwd()
    os.chdir(data_dir)
    Database = importlib.import_module('database').Database
    os.chdir(cwd)
    database = Database()
    return database

if __name__ == "__main__":
    dialogues, ontology = load_dataset('multiwoz21')
    database = load_database('multiwoz21')
    res = database.query("train", [['departure', 'cambridge'], ['destination','peterborough'], ['day', 'tuesday'], ['arrive by', '11:15']], topk=3)
    print(res[0], len(res))
