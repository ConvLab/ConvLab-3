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
import glob


descriptions = {
    'movie': 'Book movie tickets for the user',
    'name.movie': 'Name of the movie, e.g. Joker, Parasite, The Avengers',
    'name.theater': 'Name of the theater, e.g. Century City, AMC Mercado 20',
    'num.tickets': 'Number of tickets, e.g. two, me and my friend, John and I',
    'time.preference': 'Preferred time or range, e.g. around 2pm, later in the evening, 4:30pm',
    'time.showing': 'The showtimes published by the theater, e.g. 5:10pm, 8:30pm',
    'date.showing': 'the date or day of the showing, e.g. today, tonight, tomrrow, April 12th.',
    'location': 'The city, or city and state, zip code and sometimes more specific regions, e.g. downtown',
    'type.screening': 'IMAX, Dolby, 3D, standard, or similar phrases for technology offerings',
    'seating': 'Various phrases from specific "row 1" to "near the back", "on an aisle", etc.',
    'date.release': 'Movie attribute published for the official movie release date.',
    'price.ticket': 'Price per ticket',
    'price.total': 'The total for the purchase of all tickets',
    'name.genre': 'Includes a wide range from classic genres like action, drama, etc. to categories like "slasher" or series like Marvel or Harry Potter',
    'description.plot': 'The movie synopsis or shorter description',
    'description.other': 'Any other movie description that is not captured by genre, name, plot.',
    'duration.movie': 'The movie runtime, e.g. 120 minutes',
    'name.person': 'Names of actors, directors, producers but NOT movie characters',
    'name.character': 'Character names like James Bond, Harry Potter, Wonder Woman',
    'review.audience': 'The audience review',
    'review.critic': 'Critic reviews like those from Rotten Tomatoes, IMDB, etc.',
    'rating.movie': 'G, PG, PG-13, R, etc.',
}

anno2slot = {
    "movie": {
        "description.other": False,  # transform to binary dialog act
        "description.plot": False,  # too long, 19 words in avg. transform to binary dialog act
    }
}


def format_turns(ori_turns):
    # delete invalid turns and merge continuous turns
    new_turns = []
    previous_speaker = None
    utt_idx = 0
    for i, turn in enumerate(ori_turns):
        speaker = 'system' if turn['speaker'].upper() == 'ASSISTANT' else 'user'
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
            raise FileNotFoundError(f'cannot find original data {original_data_zip} in tm3/, should manually download master.zip from https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)

    ontology = {'domains': {},
                'intents': {
                    'inform': {'description': 'inform the value of a slot or general information.'}
                },
                'binary_dialogue_acts': set(),
                'state': {}}
    global descriptions
    global anno2slot
    ori_ontology = json.load(open(os.path.join(original_data_dir, "TM-3-2020/ontology/entities.json")))
    assert len(ori_ontology) == 1
    domain = list(ori_ontology.keys())[0]
    domain_ontology = ori_ontology[domain]
    ontology['domains'][domain] = {'description': descriptions[domain], 'slots': {}}
    ontology['state'][domain] = {}
    for slot in domain_ontology['required']+domain_ontology['optional']:
        ontology['domains'][domain]['slots'][slot] = {
            'description': descriptions[slot],
            'is_categorical': False,
            'possible_values': [],
        }
        if slot not in anno2slot[domain]:
            ontology['state'][domain][slot] = ''
    
    dataset = 'tm3'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}
    for data_file in tqdm(glob.glob(os.path.join(original_data_dir, f"TM-3-2020/data/*.json")), desc='processing taskmaster-{}'.format(domain)):
        data = json.load(open(data_file))
        # random split, train:validation:test = 8:1:1
        random.seed(42)
        dial_ids = list(range(len(data)))
        random.shuffle(dial_ids)
        dial_id2split = {}
        for dial_id in dial_ids[:int(0.8*len(dial_ids))]:
            dial_id2split[dial_id] = 'train'
        for dial_id in dial_ids[int(0.8*len(dial_ids)):int(0.9*len(dial_ids))]:
            dial_id2split[dial_id] = 'validation'
        for dial_id in dial_ids[int(0.9*len(dial_ids)):]:
            dial_id2split[dial_id] = 'test'

        for dial_id, d in enumerate(data):
            # delete empty dialogs and invalid dialogs
            if len(d['utterances']) == 0:
                continue
            if len(set([t['speaker'] for t in d['utterances']])) == 1:
                continue
            data_split = dial_id2split[dial_id]
            dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
            cur_domains = [domain]
            goal = {
                'description': d['instructions'],
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
                        assert len(['annotations']) == 1
                        item = segment['annotations'][0]
                        intent = 'inform'  # default intent
                        slot = item['name'].strip()
                        assert slot in ontology['domains'][domain]['slots']
                        if slot in anno2slot[domain]:
                            # binary dialog act
                            turn['dialogue_acts']['binary'].append({
                                'intent': intent,
                                'domain': domain,
                                'slot': slot,
                                'value': ''
                            })
                            continue
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
                    da_tuple = (da['intent'], da['domain'], da['slot'], da['value'],)
                    bdas.add(da_tuple)
                    if da_tuple not in ontology['binary_dialogue_acts']:
                        ontology['binary_dialogue_acts'].add(da_tuple)
                turn['dialogue_acts']['binary'] = [{'intent':bda[0],'domain':bda[1],'slot':bda[2],'value':bda[3]} for bda in sorted(bdas)]

                for da in turn['dialogue_acts']['non-categorical']:
                    slot, value = da['slot'], da['value']
                    assert slot in prev_state[domain], print(da)
                    prev_state[domain][slot] = value
                
                if speaker == 'user':
                    turn['state'] = copy.deepcopy(prev_state)
                else:
                    turn['db_results'] = {}
                    if 'apis' in turns[utt_idx-1]:
                        turn['db_results'].setdefault(domain, [])
                        apis = turns[utt_idx-1]['apis']
                        turn['db_results'][domain] += apis

                dialogue['turns'].append(turn)
            dialogues_by_split[data_split].append(dialogue)

    ontology['binary_dialogue_acts'] = [{'intent':bda[0],'domain':bda[1],'slot':bda[2],'value':bda[3]} for bda in sorted(ontology['binary_dialogue_acts'])]
    dialogues = dialogues_by_split['train']+dialogues_by_split['validation']+dialogues_by_split['test']
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    rmtree(original_data_dir)
    rmtree(new_data_dir)
    return dialogues, ontology

if __name__ == '__main__':
    preprocess()
