import json
import os
from zipfile import ZipFile, ZIP_DEFLATED
import random
import json_lines
from collections import Counter
from shutil import rmtree


def preprocess():
    random.seed(42)

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

    dataset = 'metalwoz'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split: [] for split in splits}
    ZipFile('metalwoz-test-v1.zip').extract('dstc8_metalwoz_heldout.zip')
    cnt = Counter()
    for filename in ['metalwoz-v1.zip', 'dstc8_metalwoz_heldout.zip']:
        with ZipFile(filename) as zipfile:
            task_id2description = {x['task_id']: x for x in json_lines.reader(zipfile.open('tasks.txt'))}
            for path in zipfile.namelist():
                if path.startswith('dialogues'):
                    if filename == 'metalwoz-v1.zip':
                        split = random.choice(['train']*9+['validation'])
                    else:
                        split = 'test'
                    if split == 'validation':
                        print(path, split)
                    for ori_dialog in json_lines.reader(zipfile.open(path)):
                        dialogue_id = f'{dataset}-{split}-{len(dialogues_by_split[split])}'
                        domain = ori_dialog['domain']

                        task_des = task_id2description[ori_dialog['task_id']]

                        goal = {
                            'description': "user role: {}. user prompt: {}. system role: {}. system prompt: {}.".format(
                                task_des['user_role'], task_des['user_prompt'], task_des['bot_role'], task_des['bot_prompt']),
                            'inform': {},
                            'request': {}
                        }

                        dialogue = {
                            'dataset': dataset,
                            'data_split': split,
                            'dialogue_id': dialogue_id,
                            'original_id': ori_dialog['id'],
                            'domains': [domain],  # will be updated by dialog_acts and state
                            'goal': goal,
                            'turns': []
                        }

                        ontology['domains'][domain] = {
                            'description': task_des['bot_role'],
                            'slots': {}
                        }
                        cnt[ori_dialog['turns'][0]] += 1
                        # assert ori_dialog['turns'][0] == "how may I help you?", print(ori_dialog['turns'])
                        for utt_idx, utt in enumerate(ori_dialog['turns'][1:]):
                            speaker = 'user' if utt_idx % 2 == 0 else 'system'
                            turn = {
                                'speaker': speaker,
                                'utterance': utt,
                                'utt_idx': utt_idx,
                            }
                            dialogue['turns'].append(turn)

                        dialogues_by_split[split].append(dialogue)

    os.remove('dstc8_metalwoz_heldout.zip')
    new_data_dir = 'data'
    os.makedirs(new_data_dir, exist_ok=True)
    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
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
