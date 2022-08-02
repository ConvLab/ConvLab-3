import gzip
import json
from zipfile import ZipFile, ZIP_DEFLATED
import os
from shutil import rmtree
from tqdm import tqdm

def preprocess():
    original_data_dir = 'WikiDialog-OQ'
    new_data_dir = 'data'
    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'wikidialog'
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
            'original_id': item['pid'],
            'topic': item['title'],
            'turns': []
        }
        for speaker, utterance in zip(item['author_num'], item['utterances']):
            speaker = 'system' if speaker == 0 else 'user'
            turn = {
                'speaker': speaker,
                'utterance': utterance.strip(),
                'utt_idx': len(dialogue['turns']),
            }
            dialogue['turns'].append(turn)
        return dialogue
            
    data_split = 'train'
    for shard in tqdm(range(1)):
        with gzip.open(f'{original_data_dir}/data_train.jsonl-000{shard:02}-of-00099.gz','r') as fin:
            for line in fin:
                dial_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = process_dial(line, dial_id, data_split)
                dialogues_by_split[data_split].append(dialogue)

    data_split = 'validation'
    with gzip.open(f'{original_data_dir}/data_validation.jsonl.gz','r') as fin:
        for line in fin:
            dialogue = process_dial(line, dial_id, data_split)
            dialogue['dialogue_id'] = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
            dialogues_by_split[data_split].append(dialogue)
            if len(dialogues_by_split[data_split]) >= len(dialogues_by_split['train']) // 10:
                break
    
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
