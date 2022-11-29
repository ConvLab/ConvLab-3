#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 lubis <lubis@hilbert242>
#
# Distributed under terms of the MIT license.

"""
utils from AuGPT codebase
"""
import re
import os
import sys
import types
import shutil
import logging
import requests
import torch
import zipfile
import bisect
import random
import copy
import json
from collections import OrderedDict, defaultdict
from typing import Callable, Union, Set, Optional, List, Dict, Any, Tuple, MutableMapping  # noqa: 401
from dataclasses import dataclass
import pdb

DATASETS_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'augpt')
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

temp_path = os.path.dirname(os.path.abspath(__file__))
fin = open(os.path.join("/home/lubis/datasets/augpt/mapping.pair"))
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))



class AutoDatabase:
    @staticmethod
    def load(pretrained_model_name_or_path):
        database_file = os.path.join(pretrained_model_name_or_path, 'database.zip')

        with zipfile.ZipFile(database_file) as zipf:
            def _build_database():
                module = types.ModuleType('database')
                exec(zipf.read('database.py').decode('utf-8'), module.__dict__)
                return module.Database(zipf)

            database = _build_database()


        return database

class BeliefParser:
    def __init__(self):
        self.slotval_re = re.compile(r"(\w[\w ]*\w) = ([\w\d: |']+)")
        self.domain_re = re.compile(r"(\w+) {\s*([\w,= :\d|']*)\s*}", re.IGNORECASE)

    def __call__(self, raw_belief: str):
        belief = OrderedDict()
        for match in self.domain_re.finditer(raw_belief):
            domain, domain_bs = match.group(1), match.group(2)
            belief[domain] = {}
            for slot_match in self.slotval_re.finditer(domain_bs):
                slot, val = slot_match.group(1), slot_match.group(2)
                belief[domain][slot] = val
        return belief

class AutoLexicalizer:
    @staticmethod
    def load(pretrained_model_name_or_path):
        lexicalizer_file = os.path.join(pretrained_model_name_or_path, 'lexicalizer.zip')
        
        with zipfile.ZipFile(lexicalizer_file) as zipf:
            def _build_lexicalizer():
                module = types.ModuleType('lexicalizer')
                exec(zipf.read('lexicalizer.py').decode('utf-8'), module.__dict__)
                # return module.Lexicalizer(zipf)
                return Lexicalizer(zipf)

            lexicalizer = _build_lexicalizer()


        return lexicalizer

def build_blacklist(items, domains=None):
    for i, (dialogue, items) in enumerate(items):
        if domains is not None and set(dialogue['domains']).difference(domains):
            yield i
        elif items[-1]['speaker'] != 'system':
            yield i

class BlacklistItemsWrapper:
    def __init__(self, items, blacklist):
        self.items = items
        self.key2idx = items.key2idx
        self._indexmap = []
        blacklist_pointer = 0
        for i in range(len(items)):
            if blacklist_pointer >= len(blacklist):
                self._indexmap.append(i)
            elif i < blacklist[blacklist_pointer]:
                self._indexmap.append(i)
            elif i == blacklist[blacklist_pointer]:
                blacklist_pointer += 1
        assert len(self._indexmap) == len(items) - len(blacklist)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = self.key2idx[idx]
        return self.items[self._indexmap[idx]]

    def __len__(self):
        return len(self._indexmap)
def split_name(dataset_name: str):
    split = dataset_name.rindex('/')
    return dataset_name[:split], dataset_name[split + 1:]

@dataclass
class DialogDatasetItem:
    context: Union[List[str], str]
    belief: Union[Dict[str, Dict[str, str]], str] = None
    database: Union[List[Tuple[str, int]], List[Tuple[str, int, Any]], None, str] = None
    response: str = None
    positive: bool = True
    raw_belief: Any = None
    raw_response: str = None
    key: str = None

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'belief' and val is None and self.raw_belief is not None:
            val = format_belief(self.raw_belief)
            self.belief = val
        return val

@dataclass
class DialogDataset(torch.utils.data.Dataset):
    items: List[any]
    database: Any = None
    domains: List[str] = None
    lexicalizer: Any = None
    transform: Callable[[Any], Any] = None
    normalize_input: Callable[[str], str] = None
    ontology: Dict[Tuple[str, str], Set[str]] = None

    @staticmethod
    def build_dataset_without_database(items, *args, **kwargs):
        return DialogDataset(items, FakeDatabase(), *args, **kwargs)

    def __getitem__(self, index):
        item = self.items[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)

    def map(self, transformation):
        def trans(x):
            x = self.transform(x)
            x = transformation(x)
            return x
        return dataclasses.replace(self, transform=trans)

    def finish(self, progressbar: Union[str, bool] = False):
        if self.transform is None:
            return self

        ontology = defaultdict(lambda: set())
        domains = set(self.domains) if self.domains else set()

        items = []
        for i in trange(len(self),
                        desc=progressbar if isinstance(progressbar, str) else 'loading dataset',
                        disable=not progressbar):
            item = self[i]
            for k, bs in item.raw_belief.items():
                domains.add(k)
                for k2, val in bs.items():
                    ontology[(k, k2)].add(val)
            items.append(item)
        if self.ontology:
            ontology = merge_ontologies((self.ontology, ontology))
        return dataclasses.replace(self, items=items, transform=None, domains=domains, ontology=ontology)

class DialogueItems:
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, dialogues):
        lengths = [len(x['items']) for x in dialogues]
        self.keys = [x['name'] for x in dialogues]
        self.key2idx = {k:i for (i, k) in enumerate(self.keys)}
        self.cumulative_sizes = DialogueItems.cumsum(lengths)
        self.dialogues = dialogues

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dialogue_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dialogue_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dialogue_idx - 1]
        return self.dialogues[dialogue_idx], self.dialogues[dialogue_idx]['items'][:sample_idx + 1]

    def __len__(self):
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def augpt_normalize(text, delexicalize=True, online=False):
    # lower case every word
    text = text.lower()

    text = text.replace(" 1 ", " one ")

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    if delexicalize:
        text = re.sub(pricepat, ' [price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    if delexicalize and not online:
        text = re.sub('[\"\:<>@\(\)]', '', text)
    elif delexicalize and online:
        text = re.sub('[\"\<>@\(\)]', '', text)
        text = re.sub("(([^0-9]):([^0-9]|.*))|(([^0-9]|.*):([^0-9]))", "\\2\\3\\5\\6", text) #only replace colons if it's not surrounded by digits. this wasn't a problem in standalone LAVA because time is delexicalized before normalization
    else:
        text = re.sub('[\"\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def load_dataset(name, use_goal=False, context_window_size=15, domains=None, **kwargs) -> DialogDataset:
    name, split = split_name(name)
    path = os.path.join(DATASETS_PATH, name)
    with open(os.path.join(path, f'{split}.json'), 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    dialogues = data['dialogues']
    items = DialogueItems(dialogues)
    items = BlacklistItemsWrapper(items, list(build_blacklist(items, domains)))

    def transform(x):
        dialogue, items = x
        context = [s['text'] for s in items[:-1]]
        if context_window_size is not None and context_window_size > 0:
            context = context[-context_window_size:]
        belief = items[-1]['belief']
        database = items[-1]['database']
        item = DialogDatasetItem(context,
                        raw_belief=belief,
                        database=database,
                        response=items[-1]['delexicalised_text'],
                        raw_response=items[-1]['text'],
                        key=dialogue['name'])
        if use_goal:
            setattr(item, 'goal', dialogue['goal'])
            # MultiWOZ evaluation uses booked domains property
            if 'booked_domains' in items[-1]:
                setattr(item, 'booked_domains', items[-1]['booked_domains'])
            setattr(item, 'dialogue_act', items[-1]['dialogue_act'])
        setattr(item, 'active_domain', items[-1]['active_domain'])
        return item

    dataset = DialogDataset(items, transform=transform, domains=data['domains'])
    if os.path.exists(os.path.join(path, 'database.zip')):
        dataset.database = AutoDatabase.load(path)

    if os.path.exists(os.path.join(path, 'lexicalizer.zip')):
        dataset.lexicalizer = AutoLexicalizer.load(path)

    return dataset

def format_belief(belief: OrderedDict) -> str:
    assert isinstance(belief, OrderedDict)
    str_bs = []
    for domain, domain_bs in belief.items():
        domain_bs = ', '.join([f'{slot} = {val}' for slot, val in sorted(domain_bs.items(), key=lambda x: x[0])])
        str_bs.extend([domain, '{' + domain_bs + '}'])
    return ' '.join(str_bs)

class Lexicalizer:
    def __init__(self, zipf):
        self.path = zipf.filename

    placeholder_re = re.compile(r'\[(\s*[\w_\s]+)\s*\]')
    number_re = re.compile(r'.*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s$')
    time_re = re.compile(r'((?:\d{1,2}[:]\d{2,3})|(?:\d{1,2} (?:am|pm)))', re.IGNORECASE)

    @staticmethod
    def ends_with_number(s):
        return bool(Lexicalizer.number_re.match(s))

    @staticmethod
    def extend_database_results(database_results, belief):
        # Augment database results from the belief state
        database_results = OrderedDict(database_results)
        if belief is not None:
            for i, (domain, (num_results, results)) in enumerate(database_results.items()):
                if domain not in belief:
                    continue
                if num_results == 0:
                    database_results[domain] = (1, [belief[domain]])
                else:
                    new_results = []
                    for r in results:
                        r = dict(**r)
                        for k, val in belief[domain].items():
                            if k not in r:
                                r[k] = val
                        new_results.append(r)
                    database_results[domain] = (num_results, new_results)
        return database_results

    @staticmethod
    def extend_empty_database(database_results, belief):
        # Augment database results from the belief state
        database_results = OrderedDict(database_results)
        if belief is not None:
            for domain in belief.keys():
                if domain not in database_results.keys():
                    if any([len(v) > 0 for v in belief[domain]["semi"].values()] + [len(v) > 0 for v in belief[domain]["book"].values()]):
                        database_results[domain] = (1, [belief[domain]])

        return database_results

    def __call__(self, text, database_results, belief=None, context=None):
        database_results = Lexicalizer.extend_database_results(database_results, belief)
        database_results =  Lexicalizer.extend_empty_database(database_results, belief)
        result_index = 0
        last_assignment = defaultdict(set)

        def trans(label, span, force=False, loop=100):
            nonlocal result_index
            nonlocal last_assignment
            result_str = None

            for domain, (count, results) in database_results.items():
                if domain in ["hotel", "attraction"] and label == "price":
                    label = "price range"

                # if count == 0:
                    # pdb.set_trace()
                    # # continue
                    # if label in result['semi']:
                        # result_str = result['semi'][label]
                    # elif label is result['book']:
                        # result_str = result['book'][label]
                # else:
                if domain == "train" and "arrive by" in results[0]["semi"]:
                    result = results[-1]
                else:
                    result = results[result_index % len(results)]
                # if domain == "train" and label == "id":
                #     label = "trainID"
                if label in result:
                    result_str = result[label]
                    if result_str == '?':
                        result_str = 'unknown'
                    if label == 'price range' and result_str == 'moderate' and \
                            not text[span[1]:].startswith(' price range') and \
                            not text[span[1]:].startswith(' in price'):
                        result_str = 'moderately priced'
                elif label in result['book']:
                    result_str = result['book'][label]
                elif label in result['semi']:
                    result_str = result['semi'][label]
                
                # if label == 'type':
                    # pdb.set_trace()
                    # if text[:span[0]].endswith('no ') or text[:span[0]].endswith('any ') or \
                            # text[:span[0]].endswith('some ') or Lexicalizer.ends_with_number(text[:span[0]]):
                        # if not result_str.endswith('s'):
                            # result_str += 's'
                if label == 'time' and ('[leave at]' in text or '[arrive by]' in text) and \
                    belief is not None and 'train' in belief and \
                        any([k in belief['train'] for k in ('leave at', 'arrive by')]):
                    # this is a specific case in which additional [time] slot needs to be lexicalised
                    # directly from the belief state
                    # "The earliest train after [time] leaves at ... and arrives by ..."
                    if 'leave at' in belief['train']:
                        result_str = belief['train']['leave at']
                    else:
                        result_str = belief['train']['arrive by']
                # elif label == 'time' and 'restaurant' in belief and 'book' in belief['restaurant']:
                        # result_str = belief['restaurant']['book']['time']
                elif label == 'count':
                    result_str = str(count)
                elif label == 'price' and domain == "train" and "total" in text[:span[0]]: 
                    try:
                        num_people = int(result['book']['people'])
                    except:
                        num_people = 1
                    try:
                        result_str = str(float(result[label].split()[0]) * num_people) + " pounds"
                    except:
                        result_str = ""
                elif force:
                    if label == 'time':
                        if 'leave at' in result or 'arrive by' in result:
                            if 'arrive' in text and 'arrive by' in result:
                                result_str = result['arrive by'].lstrip('0')
                            elif 'leave at' in result:
                                result_str = result['leave at'].lstrip('0')
                        elif context is not None and len(context) > 0:
                            last_utt = context[-1]
                            mtch = Lexicalizer.time_re.search(last_utt)
                            if mtch is not None:
                                result_str = mtch.group(1).lstrip('0')
                    elif label == 'name':
                        result_str = "the " + domain
                # if result_str == "not mentioned":
                    # pdb.set_trace()
                if result_str is not None:
                    break
            if force and result_str is None:
                # for domains with no database or cases with failed database search
                # if domain == "hospital":
                #     if label == 'name':
                        # result_str = "Addenbrookes hospital"
                    # elif label == "postcode":
                        # result_str = "cb20qq"
                    # elif label == "address":
                        # result_str = "hills rd , cambridge"
                    # elif label == "phone":
                        # result_str = "01223216297"
                # else:
                if label == 'reference':
                    result_str = 'YF86GE4J'
                elif label == 'phone':
                    result_str = '01223358966'
                elif label == 'postcode':
                    result_str = 'cb11jg'
                elif label == 'agent':
                    result_str = 'Cambridge Towninfo Centre'
                elif label == 'stars':
                    result_str = '4'
                elif label == 'car':
                    result_str = 'black honda taxi'
                elif label == 'address':
                    result_str = 'Parkside, Cambridge'
                elif label == 'name':
                    result_str = "it"

            if result_str is not None and result_str.lower() in last_assignment[label] and loop > 0:
                result_index += 1
                return trans(label, force=force, loop=loop - 1, span=span)

            if result_str is not None:
                last_assignment[label].add(result_str.lower())
            return result_str or f'[{label}]'

        text = Lexicalizer.placeholder_re.sub(lambda m: trans(m.group(1), span=m.span()), text)
        text = Lexicalizer.placeholder_re.sub(lambda m: trans(m.group(1), force=True, span=m.span()), text)
        return text, database_results

    def save(self, path):
        shutil.copy(self.path, os.path.join(path, os.path.split(self.path)[-1]))
