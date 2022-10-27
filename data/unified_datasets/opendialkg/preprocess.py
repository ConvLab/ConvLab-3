from turtle import st
from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
from collections import Counter
from pprint import pprint
import re
import requests
from dateutil import parser as date_parser
from string import punctuation
from copy import deepcopy
import csv
import random


def preprocess():
    random.seed(42)

    data_file = "opendialkg.csv"
    if not os.path.exists(data_file):
        response = requests.get("https://github.com/facebookresearch/opendialkg/raw/main/data/opendialkg.csv")
        open(data_file, "wb").write(response.content)

    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'opendialkg'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}

    ontology = {'domains': {},
                'intents': {},
                'state': {},
                'dialogue_acts': {
                    "categorical": {},
                    "non-categorical": {},
                    "binary": {}
                }}

    data = []
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            sample = {}
            for i, col in enumerate(row):
                sample[header[i]] = col
            data.append(sample)

    # shuffle for random split to train:validation:test = 70:15:15
    random.shuffle(data)
    split2range = {
        'train': [0, round(len(data)*0.7)],
        'validation': [round(len(data)*0.7), round(len(data)*0.85)],
        'test': [round(len(data)*0.85), len(data)],
    }
    cnt = 0
    for data_split in splits:
        for i in tqdm(range(*split2range[data_split])):
            item = data[i]
            dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
            dialogue = {
                'dataset': dataset,
                'data_split': data_split,
                'dialogue_id': dialogue_id,
                'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                'user_rating': eval(item['User Rating']),
                'system_rating': eval(item['Assistant Rating']),
                'turns': [],
            }

            for turn in eval(item['Messages']):
                speaker = 'user' if turn['sender'] == 'user' else 'system'
                turn_type = turn['type']
                if turn_type == 'chat':
                    assert len(turn) == 3
                    if len(dialogue['turns'])>0 and speaker == dialogue['turns'][-1]['speaker']:
                        dialogue['turns'][-1]['utterance'] += turn['message']
                    else:
                        dialogue['turns'].append({
                            'speaker': speaker,
                            'utterance': turn['message'],
                            'utt_idx': len(dialogue['turns']),
                        })
                elif turn['action_id'] == "meta_thread/send_meta_message":
                    # skip annotator communication
                    pass
                else:
                    assert turn_type == 'action' and turn['action_id'] == "kgwalk/choose_path"
                    assert len(dialogue['turns'])==0 or (speaker != dialogue['turns'][-1]['speaker']), print(turn)
                    dialogue['turns'].append({
                        'speaker': speaker,
                        'utterance': '',
                        'kg_path': {k: v for k, v in zip(['score', 'triples', 'rendering'], turn['metadata']['path'])},
                        'utt_idx': len(dialogue['turns']),
                    })
            if len(dialogue['turns']) != 0:
                dialogues_by_split[data_split].append(dialogue)
                if any(['kg_path' in turn for turn in dialogue['turns']]):
                    cnt+=1
    
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
    print(cnt, len(dialogues), cnt/len(dialogues))
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
