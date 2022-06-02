from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree
import json
import os
from tqdm import tqdm
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def preprocess():
    original_data_dir = 'original_data'

    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'personachat'
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

    detokenizer = TreebankWordDetokenizer()

    def sentence_normalize(utt):
        return ' '.join([detokenizer.detokenize(word_tokenize(s)) for s in sent_tokenize(utt.strip())])

    for data_split in splits:
        filename = data_split if data_split != 'validation' else 'valid'
        with open(f'{original_data_dir}/{filename}_both_original.txt') as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=data_split):
            line = line.strip()
            if line.startswith('1 '):
                # new dialog
                dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = {
                    'dataset': dataset,
                    'data_split': data_split,
                    'dialogue_id': dialogue_id,
                    'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                    'persona': {'user': [], 'system': []},
                    'turns': []
                }
                dialogues_by_split[data_split].append(dialogue)

            line = line.split('\t\t')
            if len(line) == 1:
                if 'your persona' in line[0]:
                    dialogue['persona']['system'].append(sentence_normalize(line[0].split('your persona: ')[1]))
                elif "partner's persona" in line[0]:
                    dialogue['persona']['user'].append(sentence_normalize(line[0].split("partner's persona: ")[1]))
                else:
                    assert 0, print(line)
            else:
                post, response = line[0].split('\t')
                post = ' '.join(post.split()[1:])
                candidates = line[1].split('|')
                dialogue['turns'].append({
                    'speaker': 'user',
                    'utterance': sentence_normalize(post),
                    'utt_idx': len(dialogue['turns']),
                })
                dialogue['turns'].append({
                    'speaker': 'system',
                    'utterance': sentence_normalize(response),
                    'utt_idx': len(dialogue['turns']),
                    'candidates': [sentence_normalize(candidate) for candidate in candidates]
                })

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
