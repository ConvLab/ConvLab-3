from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
from collections import Counter
from pprint import pprint
import random
import requests


def preprocess():
    data_file = "commongen_data.zip"
    if not os.path.exists(data_file):
        response = requests.get("https://storage.googleapis.com/huggingface-nlp/datasets/common_gen/commongen_data.zip")
        open(data_file, "wb").write(response.content)

    archive = ZipFile(data_file)

    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'commongen'
    speaker = 'system'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}

    ontology = {'domains': {},
                'intents': {},
                'state': {},
                'dialogue_acts': {
                    "categorical": [],
                    "non-categorical": [],
                    "binary": []
                }}

    data_split2suffix = {'train': 'train', 'validation': 'dev', 'test': 'test_noref'}
    random.seed(42)
    for data_split in splits:
        with archive.open(f'commongen.{data_split2suffix[data_split]}.jsonl') as f:
            for line in f:
                line = line.replace(b", }", b"}")  # Fix possible JSON format error
                item = json.loads(line)
                concepts = item["concept_set"].split("#")
                random.shuffle(concepts)
                scenes = item.get("scene", [''])
                for scene in scenes:
                    dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                    dialogue = {
                        'dataset': dataset,
                        'data_split': data_split,
                        'dialogue_id': dialogue_id,
                        'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                        'turns': [{
                            'speaker': speaker,
                            'utterance': scene.strip(),
                            'utt_idx': 0,
                            'concepts': concepts,
                        }]
                    }

                    dialogues_by_split[data_split].append(dialogue)

    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
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
