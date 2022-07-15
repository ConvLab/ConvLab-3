import gzip
import json
from zipfile import ZipFile, ZIP_DEFLATED
import os
from shutil import rmtree
from tqdm import tqdm
import io

def preprocess():
    original_data_dir = 'dstc8-reddit-corpus'
    new_data_dir = 'data'
    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'reddit'
    splits = ['train', 'validation']
    dialogues_by_split = {split:[] for split in splits}

    ontology = {
        'domains': {},
        'intents': {},
        'state': {},
        "dialogue_acts": {
            "categorical": {},
            "non-categorical": {},
            "binary": {}
        }
    }

    def process_dial(line, dial_id, data_split):
        item = json.loads(line)
        dialogue = {
            'dataset': dataset,
            'data_split': data_split,
            'dialogue_id': dial_id,
            'original_id': item['id'],
            'topic': item['domain'],
            'turns': []
        }
        for i, utterance in enumerate(item['turns']):
            if len(utterance) > 256:
                # remove dialogs that contain too long utterances
                return None
            speaker = 'system' if i % 2 == 1 else 'user'
            turn = {
                'speaker': speaker,
                'utterance': utterance.strip(),
                'utt_idx': len(dialogue['turns']),
            }
            dialogue['turns'].append(turn)
        return dialogue
            
    for data_split, filename in zip(['train', 'validation'], ['training', 'validation_date_out_domain_out']):
        with ZipFile(os.path.join(original_data_dir, f'{filename}.zip')) as zip_file:
            for file in zip_file.namelist():
                with io.TextIOWrapper(zip_file.open(file), encoding="utf-8") as f:
                    for line in f:
                        dial_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                        dialogue = process_dial(line, dial_id, data_split)
                        if dialogue:
                            dialogues_by_split[data_split].append(dialogue)
    
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    # rmtree(original_data_dir)
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
