import copy
import json
import os
from zipfile import ZipFile, ZIP_DEFLATED
from shutil import rmtree

ontology = {
    'domains': {
        'restaurant': {
            'description': 'search for a restaurant to dine',
            'slots': {
                'food': {
                    'description': 'food type of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'area': {
                    'description': 'area of the restaurant',
                    'is_categorical': True,
                    'possible_values': ["east", "west", "centre", "north", "south"]
                },
                'postcode': {
                    'description': 'postal code of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'phone': {
                    'description': 'phone number of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'address': {
                    'description': 'address of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'price range': {
                    'description': 'price range of the restaurant',
                    'is_categorical': True,
                    'possible_values': ["expensive", "moderate", "cheap"]
                },
                'name': {
                    'description': 'name of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                }
            }
        }
    },
    'intents': {
        'inform': {
            'description': 'system informs user the value of a slot'
        },
        'request': {
            'description': 'system asks the user to provide value of a slot'
        }
    },
    'state': {
        'restaurant': {
            'food': '',
            'area': '',
            'postcode': '',
            'phone': '',
            'address': '',
            'price range': '',
            'name': ''
        }
    },
    "dialogue_acts": {
        "categorical": {},
        "non-categorical": {},
        "binary": {}
    }
}


def convert_da(da, utt):
    global ontology

    converted = {
        'binary': [],
        'categorical': [],
        'non-categorical': []
    }

    for s, v in da:
        if s == 'request':
            converted['binary'].append({
                'intent': 'request',
                'domain': 'restaurant',
                'slot': v,
            })

        else:
            slot_type = 'categorical' if ontology['domains']['restaurant']['slots'][s]['is_categorical'] else 'non-categorical'

            v = v.strip()
            if v != 'dontcare' and ontology['domains']['restaurant']['slots'][s]['is_categorical']:
                if v == 'center':
                    v = 'centre'
                elif v == 'east side':
                    v = 'east'
                assert v in ontology['domains']['restaurant']['slots'][s]['possible_values'], print([s,v, utt])

            converted[slot_type].append({
                'intent': 'inform',
                'domain': 'restaurant',
                'slot': s,
                'value': v
            })

            if slot_type == 'non-categorical' and v != 'dontcare':

                start = utt.lower().find(v)

                if start != -1:
                    end = start + len(v)
                    converted[slot_type][-1]['start'] = start
                    converted[slot_type][-1]['end'] = end
                    converted[slot_type][-1]['value'] = utt[start:end]

    return converted


def preprocess():
    original_data_dir = 'woz'
    new_data_dir = 'data'
    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'woz'
    splits = ['train', 'validation', 'test']
    domain = 'restaurant'
    dialogues_by_split = {split: [] for split in splits}
    global ontology
    
    for split in splits:
        if split != 'validation':
            filename = os.path.join(original_data_dir, f'woz_{split}_en.json')
        else:
            filename = os.path.join(original_data_dir, 'woz_validate_en.json')
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f'cannot find {filename}, should manually download from https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz')

        data = json.load(open(filename))

        for item in data:
            dialogue = {
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': f'{dataset}-{split}-{len(dialogues_by_split[split])}',
                'original_id': item['dialogue_idx'],
                'domains': [domain],
                'turns': []
            }

            turns = item['dialogue']
            n_turn = len(turns)

            for i in range(n_turn):
                sys_utt = turns[i]['system_transcript'].strip()
                usr_utt = turns[i]['transcript'].strip()
                usr_da = turns[i]['turn_label']

                for s, v in usr_da:
                    if s == 'request':
                        assert v in ontology['domains']['restaurant']['slots']
                    else:
                        assert s in ontology['domains']['restaurant']['slots']

                if i != 0:
                    dialogue['turns'].append({
                        'utt_idx': len(dialogue['turns']),
                        'speaker': 'system',
                        'utterance': sys_utt,
                    })

                cur_state = copy.deepcopy(ontology['state'])
                for act_slots in turns[i]['belief_state']:
                    act, slots = act_slots['act'], act_slots['slots']
                    if act == 'inform':
                        for s, v in slots:
                            v = v.strip()
                            if v != 'dontcare' and ontology['domains']['restaurant']['slots'][s]['is_categorical']:
                                if v not in ontology['domains']['restaurant']['slots'][s]['possible_values']:
                                    if v == 'center':
                                        v = 'centre'
                                    elif v == 'east side':
                                        v = 'east'
                                    assert v in ontology['domains']['restaurant']['slots'][s]['possible_values']
                                
                            cur_state[domain][s] = v

                cur_usr_da = convert_da(usr_da, usr_utt)

                # add to dialogue_acts dictionary in the ontology
                for da_type in cur_usr_da:
                    das = cur_usr_da[da_type]
                    for da in das:
                        ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                        ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])]['user'] = True

                dialogue['turns'].append({
                    'utt_idx': len(dialogue['turns']),
                    'speaker': 'user',
                    'utterance': usr_utt,
                    'state': cur_state,
                    'dialogue_acts': cur_usr_da,
                })

            dialogues_by_split[split].append(dialogue)

    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str(
            {'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent': da[0],
             'domain': da[1], 'slot': da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(original_data_dir)
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
