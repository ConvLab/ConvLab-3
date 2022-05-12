from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
import requests


def preprocess():
    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'dart'
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

    url_prefix = "https://github.com/Yale-LILY/dart/raw/master/data/v1.1.1/"
    for data_split in splits:
        data_file = f"dart-v1.1.1-full-{data_split}.json" if data_split != 'validation' else "dart-v1.1.1-full-dev.json"
        if not os.path.exists(data_file):
            data = json.loads(requests.get(f"{url_prefix}{data_file}").content)
            json.dump(data, open(data_file, 'w'))
        else:
            # open(data_file, "wb").write(requests.get(f"{url_prefix}{data_file}").content)
            data = json.load(open(data_file))
        for item in tqdm(data, desc='processing dart-{}'.format(data_split)):
            tripleset = item["tripleset"]
            subtree_was_extended = item.get("subtree_was_extended", None)
            for annotation in item["annotations"]:
                source = annotation["source"]
                text = annotation["text"]
                if len(text) == 0:
                    continue
                ontology['domains'].setdefault(source, {'description': '', 'slots': {}})

                dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = {
                    'dataset': dataset,
                    'data_split': data_split,
                    'dialogue_id': dialogue_id,
                    'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                    'domains': [source],
                    'goal': {
                        'description': '',
                        'inform': {},
                        'request': {}
                    },
                    'turns': [{
                        'speaker': speaker,
                        'utterance': text.strip(),
                        'utt_idx': 0,
                        'dialogue_acts': {
                            'binary': [],
                            'categorical': [],
                            'non-categorical': [],
                        },
                        'tripleset': tripleset,
                        'subtree_was_extended': subtree_was_extended,
                        'db_results': {}
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
