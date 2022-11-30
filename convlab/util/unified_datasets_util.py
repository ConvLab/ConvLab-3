from copy import deepcopy
from typing import Dict, List, Tuple
from zipfile import ZipFile
import json
import os
import re
import importlib
from abc import ABC, abstractmethod
from pprint import pprint
from convlab.util.file_util import cached_path
import shutil
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm


class BaseDatabase(ABC):
    """Base class of unified database. Should override the query function."""

    def __init__(self):
        """extract data.zip and load the database."""

    @abstractmethod
    def query(self, domain: str, state: dict, topk: int, **kwargs) -> list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""


def download_unified_datasets(dataset_name, filename, data_dir):
    """
    It downloads the file of unified datasets from HuggingFace's datasets if it doesn't exist in the data directory

    :param dataset_name: The name of the dataset
    :param filename: the name of the file you want to download
    :param data_dir: the directory where the file will be downloaded to
    :return: The data path
    """
    data_path = os.path.join(data_dir, filename)
    if not os.path.exists(data_path):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        data_url = f'https://huggingface.co/datasets/ConvLab/{dataset_name}/resolve/main/{filename}'
        cache_path = cached_path(data_url)
        shutil.move(cache_path, data_path)
    return data_path


def relative_import_module_from_unified_datasets(dataset_name, filename, names2import):
    """
    It downloads a file from the unified datasets repository, imports it as a module, and returns the
    variable(s) you want from that module

    :param dataset_name: the name of the dataset, e.g. 'multiwoz21'
    :param filename: the name of the file to download, e.g. 'preprocess.py'
    :param names2import: a string or a list of strings. If it's a string, it's the name of the variable
    to import. If it's a list of strings, it's the names of the variables to import
    :return: the variable(s) that are being imported from the module.
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(
        __file__), f'../../../data/unified_datasets/{dataset_name}'))
    assert filename.endswith('.py')
    assert isinstance(names2import, str) or (
        isinstance(names2import, list) and len(names2import) > 0)
    data_path = download_unified_datasets(dataset_name, filename, data_dir)
    module_spec = importlib.util.spec_from_file_location(
        filename[:-3], data_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if isinstance(names2import, str):
        return eval(f'module.{names2import}')
    else:
        variables = []
        for name in names2import:
            variables.append(eval(f'module.{name}'))
        return variables


def load_dataset(dataset_name: str, dial_ids_order=None, split2ratio={}) -> Dict:
    """load unified dataset from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`
        dial_ids_order (int): idx of shuffled dial order in `data/unified_datasets/$dataset_name/shuffled_dial_ids.json`
        split2ratio (dict): a dictionary that maps the data split to the ratio of the data you want to use. 
            For example, if you want to use only half of the training data, you can set split2ratio = {'train': 0.5}

    Returns:
        dataset (dict): keys are data splits and the values are lists of dialogues
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(
        __file__), f'../../../data/unified_datasets/{dataset_name}'))
    data_path = download_unified_datasets(dataset_name, 'data.zip', data_dir)

    archive = ZipFile(data_path)
    with archive.open('data/dialogues.json') as f:
        dialogues = json.loads(f.read())
    dataset = {}
    if dial_ids_order is not None:
        data_path = download_unified_datasets(
            dataset_name, 'shuffled_dial_ids.json', data_dir)
        dial_ids = json.load(open(data_path))[dial_ids_order]
        for data_split in dial_ids:
            ratio = split2ratio.get(data_split, 1)
            dataset[data_split] = [dialogues[i]
                                   for i in dial_ids[data_split][:round(len(dial_ids[data_split])*ratio)]]
    else:
        for dialogue in dialogues:
            if dialogue['data_split'] not in dataset:
                dataset[dialogue['data_split']] = [dialogue]
            else:
                dataset[dialogue['data_split']].append(dialogue)
        for data_split in dataset:
            if data_split in split2ratio:
                dataset[data_split] = dataset[data_split][:round(
                    len(dataset[data_split])*split2ratio[data_split])]
    return dataset


def load_ontology(dataset_name: str) -> Dict:
    """load unified ontology from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        ontology (dict): dataset ontology
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(
        __file__), f'../../../data/unified_datasets/{dataset_name}'))
    data_path = download_unified_datasets(dataset_name, 'data.zip', data_dir)

    archive = ZipFile(data_path)
    with archive.open('data/ontology.json') as f:
        ontology = json.loads(f.read())
    return ontology


def load_database(dataset_name: str):
    """load database from `data/unified_datasets/$dataset_name`

    Args:
        dataset_name (str): unique dataset name in `data/unified_datasets`

    Returns:
        database: an instance of BaseDatabase
    """
    data_dir = os.path.abspath(os.path.join(os.path.abspath(
        __file__), f'../../../data/unified_datasets/{dataset_name}'))
    data_path = download_unified_datasets(
        dataset_name, 'database.py', data_dir)
    module_spec = importlib.util.spec_from_file_location('database', data_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    Database = relative_import_module_from_unified_datasets(
        dataset_name, 'database.py', 'Database')
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
        delex_utterance=False,
        use_context=False, 
        context_window_size=0, 
        terminated=False, 
        goal=False, 
        active_domains=False,
        split_to_turn=True
    ):
    """
    > This function takes in a dataset, and returns a dictionary of data splits, where each data split
    is a list of samples

    :param dataset: dataset object from `load_dataset`
    :param data_split: which split of the data to load. Can be 'train', 'validation', 'test', or 'all',
    defaults to all (optional)
    :param speaker: 'user', 'system', or 'all', defaults to all (optional)
    :param utterance: whether to include the utterance text, defaults to False (optional)
    :param dialogue_acts: whether to include dialogue acts in the data, defaults to False (optional)
    :param state: whether to include the state of the dialogue, defaults to False (optional)
    :param db_results: whether to include the database results in the context, defaults to False
    (optional)
    :param use_context: whether to include the context of the current turn in the data, defaults to
    False (optional)
    :param context_window_size: the number of previous turns to include in the context, defaults to 0
    (optional)
    :param terminated: whether to include the terminated signal, defaults to False (optional)
    :param goal: whether to include the goal of the dialogue in the data, defaults to False (optional)
    :param active_domains: whether to include the active domains of the dialogue, defaults to False
    (optional)
    :param split_to_turn: If True, each turn is a sample. If False, each dialogue is a sample, defaults
    to True (optional)
    """
    data_splits = dataset.keys() if data_split == 'all' else [data_split]
    assert speaker in ['user', 'system', 'all']
    assert not use_context or context_window_size > 0
    info_list = list(filter(eval, ['utterance', 'dialogue_acts', 'state', 'db_results', 'delex_utterance']))
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
                        sample['terminated'] = turn['utt_idx'] == len(
                            dialogue['turns']) - 1
                    if speaker == 'system' and 'booked' in turn:
                        sample['booked'] = turn['booked']
                    data_by_split[data_split].append(sample)
            if not split_to_turn:
                dialogue['turns'] = context
                data_by_split[data_split].append(dialogue)
    return data_by_split


def load_nlu_data(dataset, data_split='all', speaker='user', use_context=False, context_window_size=0, **kwargs):
    """
    It loads the data from the specified dataset, and returns it in a format that is suitable for
    training a NLU model

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'user' or 'system', defaults to user (optional)
    :param use_context: whether to use context or not, defaults to False (optional)
    :param context_window_size: the number of previous utterances to include as context, defaults to 0
    (optional)
    :return: A list of dictionaries, each dictionary contains the utterance, dialogue acts, and context.
    """
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', use_context)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('dialogue_acts', True)
    return load_unified_data(dataset, **kwargs)


def load_dst_data(dataset, data_split='all', speaker='user', context_window_size=100, **kwargs):
    """
    It loads the data from the specified dataset, with the specified data split, speaker, context window
    size, suitable for training a DST model

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'user' or 'system', defaults to user (optional)
    :param context_window_size: the number of utterances to include in the context window, defaults to
    100 (optional)
    :return: A list of dictionaries, each dictionary contains the utterance, dialogue state, and context.
    """
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('state', True)
    return load_unified_data(dataset, **kwargs)


def load_policy_data(dataset, data_split='all', speaker='system', context_window_size=1, **kwargs):
    """
    It loads the data from the specified dataset, and returns it in a format that is suitable for
    training a policy

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'system' or 'user', defaults to system (optional)
    :param context_window_size: the number of previous turns to include as context, defaults to 1
    (optional)
    :return: A list of dictionaries, each dictionary contains the utterance, dialogue state, db results, 
    dialogue acts, terminated, and context.
    """
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
    """
    It loads the data from the specified dataset, and returns it in a format that is suitable for
    training a NLG model

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'system' or 'user', defaults to system (optional)
    :param use_context: whether to use context (i.e. previous utterances), defaults to False (optional)
    :param context_window_size: the number of previous utterances to include as context, defaults to 0
    (optional)
    :return: A list of dictionaries, each dictionary contains the utterance, dialogue acts, and context
    """
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', use_context)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    kwargs.setdefault('dialogue_acts', True)
    return load_unified_data(dataset, **kwargs)


def load_e2e_data(dataset, data_split='all', speaker='system', context_window_size=100, **kwargs):
    """
    It loads the data from the specified dataset, and returns it in a format that is suitable for
    training an End2End model

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'system' or 'user', defaults to system (optional)
    :param context_window_size: the number of utterances to include in the context window, defaults to
    100 (optional)
    :return: A list of dictionaries, each dictionary contains the utterance, state, db results, 
    dialogue acts, and context
    """
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
    """
    It loads the data from the dataset, and returns it in a format that is suitable for training a 
    response generation model

    :param dataset: dataset object from `load_dataset`
    :param data_split: 'train', 'validation', 'test', or 'all', defaults to all (optional)
    :param speaker: 'system' or 'user', defaults to system (optional)
    :param context_window_size: the number of words to include in the context window, defaults to 100
    (optional)
    :return: A list of dictionaries, each dictionary contains the utterance and context
    """
    kwargs.setdefault('data_split', data_split)
    kwargs.setdefault('speaker', speaker)
    kwargs.setdefault('use_context', True)
    kwargs.setdefault('context_window_size', context_window_size)
    kwargs.setdefault('utterance', True)
    return load_unified_data(dataset, **kwargs)


def create_delex_data(dataset, delex_func=lambda d, s, v: f'[({d})-({s})]', ignore_values=['yes', 'no']):
    """add delex_utterance to the dataset according to dialogue acts and belief_state
    delex_func: function that return the placeholder (e.g. "[(domain_name)-(slot_name)]") given (domain, slot, value)
    ignore_values: ignored values when delexicalizing using the categorical acts and states
    """
    def delex_inplace(texts_placeholders, value_pattern):
        """
        It takes a list of strings and placeholders, and a regex pattern. If the pattern matches exactly
        one string, it replaces that string with a placeholder and returns True. Otherwise, it returns
        False

        :param texts_placeholders: a list of tuples, each tuple is a string and a boolean. The boolean
        indicates whether the string is a placeholder or not
        :param value_pattern: a regular expression that matches the value to be delexicalized
        :return: A list of tuples. Each tuple contains a string and a boolean. The string is either a
        placeholder or a piece of text. The boolean is True if the string is a placeholder, False
        otherwise.
        """
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
            texts_placeholders[idx:idx+1] = [
                (substring[0:start], False), (placeholder, True), (substring[end:], False)]
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
                spans = [x for x in turn['dialogue_acts']
                         ['non-categorical'] if 'start' in x]
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
                        pattern = re.compile(
                            r'\b({})\b'.format(value), flags=re.I)
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
                                    #TODO: value = ?
                                    value = '\?' if value == '?' else value
                                    try:
                                        pattern = re.compile(r'\b({})\b'.format(value), flags=re.I)
                                    except Exception:
                                        print(value)
                                    if delex_inplace(delex_utt, pattern):
                                        delex_vocab.add(placeholder)

                turn['delex_utterance'] = ''.join([x[0] for x in delex_utt])

    return dataset, sorted(list(delex_vocab))


def retrieve_utterances(query_turns, turn_pool, top_k, model_name):
    """
    It takes a list of query turns, a list of turn pool, and a top_k value, and returns a list of query
    turns with a new key called 'retrieve_utterances' that contains a list of top_k retrieved utterances
    from the turn pool
    
    :param query_turns: a list of turns that you want to retrieve utterances for
    :param turn_pool: the pool of turns to retrieve from
    :param top_k: the number of utterances to retrieve for each query turn
    :param model_name: the name of the model you want to use
    :return: A list of dictionaries, with a new key 'retrieve_utterances' that is a list of retrieved turns and similarity scores.
    """
    embedder = SentenceTransformer(model_name)
    corpus = [turn['utterance'] for turn in turn_pool]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.to('cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    queries = [turn['utterance'] for turn in query_turns]
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)
    query_embeddings = query_embeddings.to('cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=top_k)

    for i, turn in enumerate(query_turns):
        turn['retrieved_turns'] = [{'score': hit['score'], **turn_pool[hit['corpus_id']]} for hit in hits[i]]
    return query_turns


if __name__ == "__main__":
    dataset = load_dataset('multiwoz21', dial_ids_order=0)
    train_ratio = 0.1
    dataset['train'] = dataset['train'][:round(
        len(dataset['train'])*train_ratio)]
    print(len(dataset['train']))
    print(dataset.keys())
    print(len(dataset['test']))

    from convlab.util.unified_datasets_util import BaseDatabase
    database = load_database('multiwoz21')
    res = database.query("train", {'train':{'departure':'cambridge', 'destination':'peterborough', 'day':'tuesday', 'arrive by':'11:15'}}, topk=3)
    print(res[0], len(res))

    data_by_split = load_nlu_data(dataset, data_split='test', speaker='user')
    query_turns = data_by_split['test'][:10]
    pool_dataset = load_dataset('camrest')
    turn_pool = load_nlu_data(pool_dataset, data_split='train', speaker='user')['train']
    augmented_dataset = retrieve_utterances(query_turns, turn_pool, 3, 'all-MiniLM-L6-v2')
    pprint(augmented_dataset[0])

    def delex_slot(domain, slot, value):
        # only use slot name for delexicalization
        return f'[{slot}]'

    dataset, delex_vocab = create_delex_data(dataset, delex_slot)
    json.dump(dataset['test'], open('new_delex_multiwoz21_test.json',
              'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(delex_vocab, open('new_delex_vocab.json', 'w',
              encoding='utf-8'), indent=2, ensure_ascii=False)
    with open('new_delex_cmp.txt', 'w') as f:
        for dialog in dataset['test']:
            for turn in dialog['turns']:
                f.write(turn['utterance']+'\n')
                f.write(turn['delex_utterance']+'\n')
                f.write('\n')
