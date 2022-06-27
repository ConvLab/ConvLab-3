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


def value_in_utt(value, utt):
    """return character level (start, end) if value in utt"""
    value = value.strip(punctuation).lower()
    utt = utt
    p = '(^|[\s,\.:\?!-])(?P<v>{})([\s,\.:\?!-\']|$)'.format(re.escape(value))
    p = re.compile(p, re.I)
    m = re.search(p, utt)
    if m:
        # very few value appears more than once, take the first span
        return True, m.span('v')
    else:
        try:
            # solve date representation, e.g. '3 pm' vs '3pm'
            date_parser.parse(value)
            if (value.endswith('pm') or value.endswith('am')) and ''.join(value.split(' ')) in ''.join(utt.split(' ')):
                return True, None
            
        except:
            if value in utt:
                # value appears, but may be in the plural, -ing, -ly, etc.
                return True, None

    return False, None


def preprocess():
    data_file = "kvret_dataset_public.zip"
    if not os.path.exists(data_file):
        response = requests.get("http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip")
        open(data_file, "wb").write(response.content)

    archive = ZipFile(data_file)

    new_data_dir = 'data'

    os.makedirs(new_data_dir, exist_ok=True)

    dataset = 'kvret'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}

    ontology = {'domains': {},
                'intents': {
                    'inform': {'description': ''},
                    'request': {'description': ''}
                },
                'state': {},
                'dialogue_acts': {
                    "categorical": {},
                    "non-categorical": {},
                    "binary": {}
                }}

    domain2slot = {
        'schedule': ['event', 'time', 'date', 'party', 'room', 'agenda'],
        'weather': ['location', 'weekly_time', 'temperature', 'weather_attribute'],
        'navigate': ['poi', 'traffic_info', 'poi_type', 'address', 'distance']
    }
    slot2domain = {slot: domain for domain in domain2slot for slot in domain2slot[domain]}

    db = []
    with archive.open(f'kvret_entities.json') as f:
        entities = json.load(f)
        for slot, values in entities.items():
            domain = slot2domain[slot]
            ontology['domains'].setdefault(domain, {'description': '', 'slots': {}})
            if slot == 'poi':
                for s in ['poi', 'address', 'poi_type']:
                    ontology['domains'][domain]['slots'][s] = {'description': '', 'is_categorical': False, 'possible_values': []}
                for item in values:
                    poi, address, poi_type = item['poi'], item['address'], item['type']
                    db.append({'poi': poi, 'address': address, 'poi_type': poi_type})
                    for s in ['poi', 'address', 'poi_type']:
                        ontology['domains'][domain]['slots'][s]['possible_values'].append(db[-1][s])
                continue
            elif slot == 'weekly_time':
                slot = 'date'
            elif slot == 'temperature':
                values = [f"{x}F" for x in values]
            elif slot == 'distance':
                values = [f"{x} miles" for x in values]
            
            ontology['domains'][domain]['slots'][slot] = {'description': '', 'is_categorical': False, 'possible_values': values}

    for domain in ontology['domains']:
        for slot in ontology['domains'][domain]['slots']:
            ontology['domains'][domain]['slots'][slot]['possible_values'] = sorted(list(set(ontology['domains'][domain]['slots'][slot]['possible_values'])))

    for data_split in splits:
        filename = data_split if data_split != 'validation' else 'dev'
        with archive.open(f'kvret_{filename}_public.json') as f:
            data = json.load(f)
            for item in tqdm(data):
                if len(item['dialogue']) == 0:
                    continue
                scenario = item['scenario']
                domain = scenario['task']['intent']

                slots = scenario['kb']['column_names']
                db_results = {domain: []}
                if scenario['kb']['items']:
                    for entry in scenario['kb']['items']:
                        db_results[domain].append({s: entry[s] for s in slots})
                
                dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
                dialogue = {
                    'dataset': dataset,
                    'data_split': data_split,
                    'dialogue_id': dialogue_id,
                    'original_id': f'{data_split}-{len(dialogues_by_split[data_split])}',
                    'domains': [domain],
                    'turns': []
                }
                init_state = {domain: {}}

                for turn in item['dialogue']:
                    speaker = 'user' if turn['turn'] == 'driver' else 'system'
                    utt = turn['data']['utterance'].strip()
                    if len(dialogue['turns']) > 0 and speaker == dialogue['turns'][-1]['speaker']:
                        # repeat, skip
                        if utt == dialogue['turns'][-1]['utterance']:
                            continue
                        else:
                            dialogue['turns'].pop(-1)

                    dialogue['turns'].append({
                        'speaker': speaker,
                        'utterance': utt,
                        'utt_idx': len(dialogue['turns']),
                        'dialogue_acts': {
                            'binary': [],
                            'categorical': [],
                            'non-categorical': [],
                        },
                    })
                    
                    if speaker == 'user':
                        dialogue['turns'][-1]['state'] = deepcopy(init_state)
                    else:
                        user_da = {'binary': [], 'categorical': [], 'non-categorical': []}
                        user_utt = dialogue['turns'][-2]['utterance']

                        for slot, value in turn['data']['slots'].items():
                            value = value.strip()
                            is_appear, span = value_in_utt(value, user_utt)
                            if is_appear:
                                if span:
                                    start, end = span
                                    user_da['non-categorical'].append({
                                        'intent': 'inform', 'domain': domain, 'slot': slot, 'value': user_utt[start:end],
                                        'start': start, 'end': end
                                    })
                                else:
                                    user_da['non-categorical'].append({
                                        'intent': 'inform', 'domain': domain, 'slot': slot, 'value': value,
                                    })
                            init_state[domain][slot] = value
                            ontology['state'].setdefault(domain, {})
                            ontology['state'][domain].setdefault(slot, '')
                        dialogue['turns'][-2]['state'] = deepcopy(init_state)
                        
                        for slot, present in turn['data']['requested'].items():
                            if slot not in turn['data']['slots'] and present:
                                user_da['binary'].append({'intent': 'request', 'domain': domain, 'slot': slot})
                        
                        dialogue['turns'][-2]['dialogue_acts'] = user_da
                        dialogue['turns'][-1]['db_results'] = db_results

                        for da_type in user_da:
                            das = user_da[da_type]
                            for da in das:
                                ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                                ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])]['user'] = True

                        assert all([s in ontology['domains'][domain]['slots'] for s in turn['data']['requested']]), print(turn['data']['requested'], ontology['domains'][domain]['slots'].keys())
                        assert all([s in ontology['domains'][domain]['slots'] for s in turn['data']['slots']]), print(turn['data']['slots'], ontology['domains'][domain]['slots'].keys())

                dialogues_by_split[data_split].append(dialogue)

    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str({'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent':da[0],'domain':da[1], 'slot':da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(db, open(f'{new_data_dir}/db.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(new_data_dir)
    return dialogues, ontology


if __name__ == '__main__':
    preprocess()
