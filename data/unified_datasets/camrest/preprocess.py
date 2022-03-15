import zipfile
import json
import os
import copy
from shutil import copy2, rmtree
from zipfile import ZipFile, ZIP_DEFLATED

ontology = {
    'domains': {
        'restaurant': {
            'description': 'find a restaurant to eat',
            'slots': {
                'area': {
                    'description': 'area where the restaurant is located',
                    'is_categorical': True,
                    'possible_values': ["centre","north","west","south","east"]
                },
                'price range': {
                    'description': 'price range of the restaurant',
                    'is_categorical': True,
                    'possible_values': ["cheap","moderate","expensive"]
                },
                'food': {
                    'description': 'the cuisine of the restaurant',
                    'is_categorical': False,
                    'possible_values': ["afghan","african","afternoon tea","asian oriental","australasian","australian","austrian","barbeque","basque","belgian","bistro","brazilian","british","canapes","cantonese","caribbean","catalan","chinese","christmas","corsica","creative","crossover","cuban","danish","eastern european","english","eritrean","european","french","fusion","gastropub","german","greek","halal","hungarian","indian","indonesian","international","irish","italian","jamaican","japanese","korean","kosher","latin american","lebanese","light bites","malaysian","mediterranean","mexican","middle eastern","modern american","modern eclectic","modern european","modern global","molecular gastronomy","moroccan","new zealand","north african","north american","north indian","northern european","panasian","persian","polish","polynesian","portuguese","romanian","russian","scandinavian","scottish","seafood","singaporean","south african","south indian","spanish","sri lankan","steakhouse","swedish","swiss","thai","the americas","traditional","turkish","tuscan","unusual","vegetarian","venetian","vietnamese","welsh","world"]
                },
                'name': {
                    'description': 'name of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'phone': {
                    'description': 'phone number of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'address': {
                    'description': 'exact location of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                },
                'postcode': {
                    'description': 'postcode of the restaurant',
                    'is_categorical': False,
                    'possible_values': []
                }
            }
        }
    },
    'intents': {
        'inform': {
            'description': 'inform the value of a slot'
        },
        'request': {
            'description': 'ask for the value of a slot'
        },
        'nooffer': {
            'description': 'inform the user that there is no result satisfies user requirements'
        }
    },
    'state': {
        'restaurant': {
            'price range': '',
            'area': '',
            'food': ''
        }
    },
    'dialogue_acts': {
        "categorical": {},
        "non-categorical": {},
        "binary": {}
    }
}


def convert_da(utt, da):
    global ontology
    converted_da = {
        'binary': [],
        'categorical': [],
        'non-categorical': []
    }

    for intent, svs in da.items():
        assert intent in ontology['intents']
        if intent == 'nooffer':
            assert svs == [['none', 'none']]
            converted_da['binary'].append({
                'intent': intent,
                'domain': 'restaurant',
                'slot': '',
            })
            continue

        for s, v in svs:
            if 'care' in v:
                assert v == 'dontcare', print(v)
            assert s == s.lower()
            if s == 'pricerange':
                s = 'price range'
            v = v
            if intent == 'request':
                assert v == '?'
                converted_da['binary'].append({
                    'intent': intent,
                    'domain': 'restaurant',
                    'slot': s
                })
                continue

            if s in ['price range', 'area']:
                assert v.lower() in ontology['domains']['restaurant']['slots'][s]['possible_values'] + ['dontcare'], print(s, v)
                converted_da['categorical'].append({
                    'intent': intent,
                    'domain': 'restaurant',
                    'slot': s,
                    'value': v
                })

            else:
                # non-categorical
                start_ch = utt.lower().find(v.lower())

                if start_ch == -1:
                    if not v == 'dontcare':
                        print('non-categorical slot value not found')
                        print('value: {}'.format(v))
                        print('sentence: {}'.format(utt))
                        print()

                    converted_da['non-categorical'].append({
                        'intent': intent,
                        'domain': 'restaurant',
                        'slot': s,
                        'value': v,
                    })
                else:
                    converted_da['non-categorical'].append({
                        'intent': intent,
                        'domain': 'restaurant',
                        'slot': s,
                        'value': utt[start_ch: start_ch + len(v)],
                        'start': start_ch,
                        'end': start_ch + len(v)
                    })
                    assert utt[start_ch: start_ch + len(v)].lower() == v.lower()

    return converted_da


def convert_state(slu):
    global ontology
    ret_state = copy.deepcopy(ontology['state'])
    for da in slu:
        if da['act'] != 'inform':
            continue

        for s, v in da['slots']:
            s = s if s != 'pricerange' else 'price range'
            if s not in ret_state['restaurant']:
                print('slot not in state')
                print(da)
                print()
                continue
            ret_state['restaurant'][s] = v

    return ret_state


def preprocess():
    # use convlab-2 version camrest which already has dialog act annotation
    original_data_dir = '../../camrest/'
    new_data_dir = 'data'
    
    os.makedirs(new_data_dir, exist_ok=True)

    copy2(f'{original_data_dir}/db/CamRestDB.json', new_data_dir)
    
    dataset = 'camrest'
    domain = 'restaurant'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}
    
    for split in ['train', 'val', 'test']:
        data = json.load(zipfile.ZipFile(os.path.join(original_data_dir, f'{split}.json.zip'), 'r').open(f'{split}.json'))
        if split == 'val':
            split = 'validation'

        cur_domains = [domain]

        for ori_dialog in data:
            dialogue_id = f'{dataset}-{split}-{len(dialogues_by_split[split])}'

            goal = {
                'description': ori_dialog['goal']['text'],
                'inform': {'restaurant': {}},
                'request': {'restaurant': {}}
            }
            for slot, value in ori_dialog['goal']['info'].items():
                if slot == 'pricerange':
                    slot = 'price range'
                goal['inform'][domain][slot] = value
            for slot in ori_dialog['goal']['reqt']:
                if slot == 'pricerange':
                    slot = 'price range'
                goal['request'][domain][slot] = ''

            dialogue = {
                'dataset': dataset,
                'data_split': split,
                'dialogue_id': dialogue_id,
                'original_id': ori_dialog['dialogue_id'],
                'domains': cur_domains,
                'goal': goal,
                'finished': ori_dialog['finished'],
                'turns': []
            }

            for turn in ori_dialog['dial']:
                usr_text = turn['usr']['transcript']
                usr_da = turn['usr']['dialog_act']

                sys_text = turn['sys']['sent']
                sys_da = turn['sys']['dialog_act']

                cur_state = convert_state(turn['usr']['slu'])
                cur_user_da = convert_da(usr_text, usr_da)

                usr_turn = {
                    'speaker': 'user',
                    'utterance': usr_text,
                    'utt_idx': len(dialogue['turns']),
                    'dialogue_acts': cur_user_da,
                    'state': cur_state,
                }

                sys_turn = {
                    'speaker': 'system',
                    'utterance': sys_text,
                    'utt_idx': len(dialogue['turns'])+1,
                    'dialogue_acts': convert_da(sys_text, sys_da),
                    'db_results': {}
                }

                dialogue['turns'].append(usr_turn)
                dialogue['turns'].append(sys_turn)

            for turn in dialogue['turns']:
                speaker = turn['speaker']
                dialogue_acts = turn['dialogue_acts']

                # add to dialogue_acts dictionary in the ontology
                for da_type in dialogue_acts:
                    das = dialogue_acts[da_type]
                    for da in das:
                        ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                        ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])][speaker] = True
            dialogues_by_split[split].append(dialogue)

    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str({'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent':da[0],'domain':da[1], 'slot':da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
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
