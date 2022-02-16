from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import copy
import zipfile
from tqdm import tqdm
import re
from collections import Counter
from shutil import rmtree
from convlab2.util.file_util import read_zipped_json, write_zipped_json
from pprint import pprint
import random


descriptions = {
    "uber_lyft": {
        "uber_lyft": "order a car for a ride inside a city",
        "location.from": "pickup location",
        "location.to": "destination of the ride",
        "type.ride": "type of ride",
        "num.people": "number of people",
        "price.estimate": "estimated cost of the ride",
        "duration.estimate": "estimated duration of the ride",
        "time.pickup": "time of pickup",
        "time.dropoff": "time of dropoff",
    },
    "movie_ticket": {
        "movie_ticket": "book movie tickets for a film",
        "name.movie": "name of the movie",
        "name.theater": "name of the theater",
        "num.tickets": "number of tickets",
        "time.start": "start time of the movie",
        "location.theater": "location of the theater",
        "price.ticket": "price of the ticket",
        "type.screening": "type of the screening",
        "time.end": "end time of the movie",
        "time.duration": "duration of the movie",
    },
    "restaurant_reservation": {
        "restaurant_reservation": "searching for a restaurant and make reservation",
        "name.restaurant": "name of the restaurant",
        "name.reservation": "name of the person who make the reservation",
        "num.guests": "number of guests",
        "time.reservation": "time of the reservation",
        "type.seating": "type of the seating",
        "location.restaurant": "location of the restaurant",
    },
    "coffee_ordering": {
        "coffee_ordering": "order a coffee drink from either Starbucks or Peets for pick up",
        "location.store": "location of the coffee store",
        "name.drink": "name of the drink",
        "size.drink": "size of the drink",
        "num.drink": "number of drinks",
        "type.milk": "type of the milk",
        "preference": "user preference of the drink",
    },
    "pizza_ordering": {
        "pizza_ordering": "order a pizza",
        "name.store": "name of the pizza store",
        "name.pizza": "name of the pizza",
        "size.pizza": "size of the pizza",
        "type.topping": "type of the topping",
        "type.crust": "type of the crust",
        "preference": "user preference of the pizza",
        "location.store": "location of the pizza store",
    },
    "auto_repair": {
        "auto_repair": "set up an auto repair appointment with a repair shop",
        "name.store": "name of the repair store",
        "name.customer": "name of the customer",
        "date.appt": "date of the appointment",
        "time.appt": "time of the appointment",
        "reason.appt": "reason of the appointment",
        "name.vehicle": "name of the vehicle",
        "year.vehicle": "year of the vehicle",
        "location.store": "location of the repair store",
    }
}

def normalize_domain_name(domain):
    if domain == 'auto':
        return 'auto_repair'
    elif domain == 'pizza':
        return 'pizza_ordering'
    elif domain == 'coffee':
        return 'coffee_ordering'
    elif domain == 'uber':
        return 'uber_lyft'
    elif domain == 'restaurant':
        return 'restaurant_reservation'
    elif domain == 'movie':
        return 'movie_ticket'
    assert 0


def format_turns(ori_turns):
    # delete invalid turns and merge continuous turns
    new_turns = []
    previous_speaker = None
    utt_idx = 0
    for i, turn in enumerate(ori_turns):
        speaker = 'system' if turn['speaker'] == 'ASSISTANT' else 'user'
        turn['speaker'] = speaker
        if turn['text'] == '(deleted)':
            continue
        if not previous_speaker:
            # first turn
            assert speaker != previous_speaker
        if speaker != previous_speaker:
            # switch speaker
            previous_speaker = speaker
            new_turns.append(copy.deepcopy(turn))
            utt_idx += 1
        else:
            # continuous speaking of the same speaker
            last_turn = new_turns[-1]
            # skip repeated turn
            if turn['text'] in ori_turns[i-1]['text']:
                continue
            # merge continuous turns
            index_shift = len(last_turn['text']) + 1
            last_turn['text'] += ' '+turn['text']
            if 'segments' in turn:
                last_turn.setdefault('segments', [])
                for segment in turn['segments']:
                    segment['start_index'] += index_shift
                    segment['end_index'] += index_shift
                last_turn['segments'] += turn['segments']
    return new_turns


def preprocess():
    original_data_dir = 'Taskmaster-master'
    new_data_dir = 'data'

    if not os.path.exists(original_data_dir):
        original_data_zip = 'master.zip'
        if not os.path.exists(original_data_zip):
            raise FileNotFoundError(f'cannot find original data {original_data_zip} in tm1/, should manually download master.zip from https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)

    ontology = {'domains': {},
                'intents': {
                    'inform': {'description': 'inform the value of a slot or general information.'},
                    'accept': {'description': 'accept the value of a slot or a transaction'},
                    'reject': {'description': 'reject the value of a slot or a transaction'}
                },
                'state': {},
                'dialogue_acts': {
                    "categorical": {},
                    "non-categorical": {},
                    "binary": {}
                }}
    global descriptions
    ori_ontology = {}
    for _, item in json.load(open(os.path.join(original_data_dir, "TM-1-2019/ontology.json"))).items():
        ori_ontology[item["id"]] = item
    
    for domain, item in ori_ontology.items():
        ontology['domains'][domain] = {'description': descriptions[domain][domain], 'slots': {}}
        ontology['state'][domain] = {}
        for slot in item['required']+item['optional']:
            ontology['domains'][domain]['slots'][slot] = {
                'description': descriptions[domain][slot],
                'is_categorical': False,
                'possible_values': [],
            }
            ontology['state'][domain][slot] = ''

    dataset = 'tm1'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}
    dialog_files = ["TM-1-2019/self-dialogs.json", "TM-1-2019/woz-dialogs.json"]
    for file_idx, filename in enumerate(dialog_files):
        data = json.load(open(os.path.join(original_data_dir, filename)))
        if file_idx == 0:
            # original split for self dialogs
            dial_id2split = {}
            for data_split in ['train', 'dev', 'test']:
                with open(os.path.join(original_data_dir, f"TM-1-2019/train-dev-test/{data_split}.csv")) as f:
                    for line in f:
                        dial_id = line.split(',')[0]
                        dial_id2split[dial_id] = data_split if data_split != 'dev' else 'validation'
        else:
            # random split for woz dialogs 8:1:1
            random.seed(42)
            dial_ids = [d['conversation_id'] for d in data]
            random.shuffle(dial_ids)
            dial_id2split = {}
            for dial_id in dial_ids[:int(0.8*len(dial_ids))]:
                dial_id2split[dial_id] = 'train'
            for dial_id in dial_ids[int(0.8*len(dial_ids)):int(0.9*len(dial_ids))]:
                dial_id2split[dial_id] = 'validation'
            for dial_id in dial_ids[int(0.9*len(dial_ids)):]:
                dial_id2split[dial_id] = 'test'

        for d in tqdm(data, desc='processing taskmaster-{}'.format(filename)):
            # delete empty dialogs and invalid dialogs
            if len(d['utterances']) == 0:
                continue
            if len(set([t['speaker'] for t in d['utterances']])) == 1:
                continue
            data_split = dial_id2split[d["conversation_id"]]
            dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
            cur_domains = [normalize_domain_name(d["instruction_id"].split('-', 1)[0])]
            assert len(cur_domains) == 1 and cur_domains[0] in ontology['domains']
            domain = cur_domains[0]
            goal = {
                'description': '',
                'inform': {},
                'request': {}
            }
            dialogue = {
                'dataset': dataset,
                'data_split': data_split,
                'dialogue_id': dialogue_id,
                'original_id': d["conversation_id"],
                'domains': cur_domains,
                'goal': goal,
                'turns': []
            }
            turns = format_turns(d['utterances'])
            prev_state = {}
            prev_state.setdefault(domain, copy.deepcopy(ontology['state'][domain]))
            
            for utt_idx, uttr in enumerate(turns):
                speaker = uttr['speaker']
                turn = {
                    'speaker': speaker,
                    'utterance': uttr['text'],
                    'utt_idx': utt_idx,
                    'dialogue_acts': {
                        'binary': [],
                        'categorical': [],
                        'non-categorical': [],
                    },
                }
                in_span = [0] * len(turn['utterance'])

                if 'segments' in uttr:
                    # sort the span according to the length
                    segments = sorted(uttr['segments'], key=lambda x: len(x['text']))
                    for segment in segments:
                        # Each conversation was annotated by two workers.
                        # only keep the first annotation for the span
                        item = segment['annotations'][0]
                        intent = 'inform'  # default intent
                        slot = item['name'].split('.', 1)[-1]
                        if slot.endswith('.accept') or slot.endswith('.reject'):
                            # intent=accept/reject
                            intent = slot[-6:]
                            slot = slot[:-7]
                        if slot not in ontology['domains'][domain]['slots']:
                            # no slot, only general reference to a transaction, binary dialog act
                            turn['dialogue_acts']['binary'].append({
                                'intent': intent,
                                'domain': domain,
                                'slot': '',
                            })
                        else:
                            assert turn['utterance'][segment['start_index']:segment['end_index']] == segment['text']
                            # skip overlapped spans, keep the shortest one
                            if sum(in_span[segment['start_index']: segment['end_index']]) > 0:
                                continue
                            else:
                                in_span[segment['start_index']: segment['end_index']] = [1]*(segment['end_index']-segment['start_index'])
                            turn['dialogue_acts']['non-categorical'].append({
                                'intent': intent,
                                'domain': domain,
                                'slot': slot,
                                'value': segment['text'],
                                'start': segment['start_index'],
                                'end': segment['end_index']
                            })

                turn['dialogue_acts']['non-categorical'] = sorted(turn['dialogue_acts']['non-categorical'], key=lambda x: x['start'])

                bdas = set()
                for da in turn['dialogue_acts']['binary']:
                    da_tuple = (da['intent'], da['domain'], da['slot'],)
                    bdas.add(da_tuple)
                turn['dialogue_acts']['binary'] = [{'intent':bda[0],'domain':bda[1],'slot':bda[2]} for bda in sorted(bdas)]
                # add to dialogue_acts dictionary in the ontology
                for da_type in turn['dialogue_acts']:
                    das = turn['dialogue_acts'][da_type]
                    for da in das:
                        ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                        ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])][speaker] = True

                for da in turn['dialogue_acts']['non-categorical']:
                    slot, value = da['slot'], da['value']
                    assert slot in prev_state[domain]
                    # not add reject slot-value into state
                    if da['intent'] != 'reject':
                        prev_state[domain][slot] = value
                
                if speaker == 'user':
                    turn['state'] = copy.deepcopy(prev_state)
                else:
                    turn['db_results'] = {}

                dialogue['turns'].append(turn)
            dialogues_by_split[data_split].append(dialogue)
    
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str({'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent':da[0],'domain':da[1], 'slot':da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
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
