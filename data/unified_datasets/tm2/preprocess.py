from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import copy
import zipfile
from tqdm import tqdm
import re
from collections import Counter
from shutil import rmtree
from convlab.util.file_util import read_zipped_json, write_zipped_json
from pprint import pprint
import random


descriptions = {
    "flights": {
        "flights": "find a round trip or multi-city flights",
        "type": "type of the flight",
        "destination1": "the first destination city of the trip",
        "destination2": "the second destination city of the trip",
        "origin": "the origin city of the trip",
        "date.depart_origin": "date of departure from origin",
        "date.depart_intermediate": "date of departure from intermediate",
        "date.return": "date of return",
        "time_of_day": "time of the flight",
        "seating_class": "seat type (first class, business class, economy class, etc.",
        "seat_location": "location of the seat",
        "stops": "non-stop, layovers, etc.",
        "price_range": "price range of the flight",
        "num.pax": "number of people",
        "luggage": "luggage information",
        "total_fare": "total cost of the trip",
        "other_description": "other description of the flight",
        "from": "departure of the flight",
        "to": "destination of the flight",
        "airline": "airline of the flight",
        "flight_number": "the number of the flight",
        "date": "date of the flight",
        "from.time": "departure time of the flight",
        "to.time": "arrival time of the flight",
        "stops.location": "location of the stop",
        "fare": "cost of the flight",
    },
    "food-ordering": {
        "food-ordering": "order take-out for a particular cuisine choice",
        "name.item": "name of the item",
        "other_description.item": "other description of the item",
        "type.retrieval": "type of the retrieval method",
        "total_price": "total price",
        "time.pickup": "pick up time",
        "num.people": "number of people",
        "name.restaurant": "name of the restaurant",
        "type.food": "type of food",
        "type.meal": "type of meal",
        "location.restaurant": "location of the restaurant",
        "rating.restaurant": "rating of the restaurant",
        "price_range": "price range of the food",
    },
    "hotels": {
        "hotels": "find a hotel using typical preferences",
        "name.hotel": "name of the hotel",
        "location.hotel": "location of the hotel",
        "sub_location.hotel": "rough location of the hotel",
        "star_rating": "star rating of the hotel",
        "customer_rating": "customer rating of the hotel",
        "customer_review": "customer review of the hotel",
        "price_range": "price range of the hotel",
        "amenity": "amenity of the hotel",
        "num.beds": "number of beds to book",
        "type.bed": "type of the bed",
        "num.rooms": "number of rooms to book",
        "check-in_date": "check-in date",
        "check-out_date": "check-out date",
        "date_range": "date range of the reservation",
        "num.guests": "number of guests",
        "type.room": "type of the room",
        "price_per_night": "price per night",
        "total_fare": "total fare",
        "location": "location of the hotel",
        "other_request": "other request",
        "other_detail": "other detail",
    },
    "movies": {
        "movies": "find a movie to watch in theaters or using a streaming service at home",
        "name.movie": "name of the movie",
        "genre": "genre of the movie",
        "name.theater": "name of the theater",
        "location.theater": "location of the theater",
        "time.start": "start time of the movie",
        "time.end": "end time of the movie",
        "price.ticket": "price of the ticket",
        "price.streaming": "price of the streaming",
        "type.screening": "type of the screening",
        "audience_rating": "audience rating",
        "critic_rating": "critic rating",
        "movie_rating": "film rating",
        "release_date": "release date of the movie",
        "runtime": "running time of the movie",
        "real_person": "name of actors, directors, etc.",
        "character": "name of character in the movie",
        "streaming_service": "streaming service that provide the movie",
        "num.tickets": "number of tickets",
        "seating": "type of seating",
        "other_description": "other description about the movie",
        "synopsis": "synopsis of the movie",
    },
    "music": {
        "music": "find several tracks to play and then comment on each one",
        "name.track": "name of the track",
        "name.artist": "name of the artist",
        "name.album": "name of the album",
        "name.genre": "music genre",
        "type.music": "rough type of the music",
        "describes_track": "description of a track to find",
        "describes_artist": "description of a artist to find",
        "describes_album": "description of an album to find",
        "describes_genre": "description of a genre to find",
        "describes_type.music": "description of the music type",
        "technical_difficulty": "there is a technical difficulty",
    },
    "restaurant-search": {
        "restaurant-search": "ask for recommendations for a particular type of cuisine",
        "name.restaurant": "name of the restaurant",
        "location": "location of the restaurant",
        "sub-location": "rough location of the restaurant",
        "type.food": "the cuisine of the restaurant",
        "menu_item": "item in the menu",
        "type.meal": "type of meal",
        "rating": "rating of the restaurant",
        "price_range": "price range of the restaurant",
        "business_hours": "business hours of the restaurant",
        "name.reservation": "name of the person who make the reservation",
        "num.guests": "number of guests",
        "time.reservation": "time of the reservation",
        "date.reservation": "date of the reservation",
        "type.seating": "type of the seating",
        "other_description": "other description of the restaurant",
        "phone": "phone number of the restaurant",
    },
    "sports": {
        "sports": "discuss facts and stats about players, teams, games, etc. in EPL, MLB, MLS, NBA, NFL",
        "name.team": "name of the team",
        "record.team": "record of the team (number of wins and losses)",
        "record.games_ahead": "number of games ahead",
        "record.games_back": "number of games behind",
        "place.team": "ranking of the team",
        "result.match": "result of the match",
        "score.match": "score of the match",
        "date.match": "date of the match",
        "day.match": "day of the match",
        "time.match": "time of the match",
        "name.player": "name of the player",
        "position.player": "position of the player",
        "record.player": "record of the player",
        "name.non_player": "name of non-palyer such as the manager, coach",
        "venue": "venue of the match take place",
        "other_description.person": "other description of the person",
        "other_description.team": "other description of the team",
        "other_description.match": "other description of the match",
    }
}

anno2slot = {
    "flights": {
        "date.depart": "date.depart_origin",  # rename
        "date.intermediate": "date.depart_intermediate",  # rename
        "flight_booked": False,  # transform to binary dialog act
    },
    "food-ordering": {
        "name.person": None,  # no sample, ignore
        "phone.restaurant": None,  # no sample, ignore
        "business_hours.restaurant": None,  # no sample, ignore
        "official_description.restaurant": None,  # 1 sample, ignore
    },
    "hotels": {
        "hotel_booked": False,  # transform to binary dialog act
    },
    "movies": {
        "time.end.": "time.end",  # rename
        "seating ticket_booking": "seating",  # mixed in the original ontology
        "ticket_booking": False,  # transform to binary dialog act
        "synopsis": False,  # too long, 54 words in avg. transform to binary dialog act
    },
    "music": {},
    "restaurant-search": {
        "offical_description": False,  # too long, 15 words in avg. transform to binary dialog act
    },
    "sports": {}
}


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
            raise FileNotFoundError(f'cannot find original data {original_data_zip} in tm2/, should manually download master.zip from https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)

    ontology = {'domains': {},
                'intents': {
                    'inform': {'description': 'inform the value of a slot or general information.'}
                },
                'state': {},
                'dialogue_acts': {
                    "categorical": {},
                    "non-categorical": {},
                    "binary": {}
                }}
    global descriptions
    global anno2slot
    domains = ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']
    for domain in domains:
        domain_ontology = json.load(open(os.path.join(original_data_dir, f"TM-2-2020/ontology/{domain}.json")))
        assert len(domain_ontology) == 1
        ontology['domains'][domain] = {'description': descriptions[domain][domain], 'slots': {}}
        ontology['state'][domain] = {}
        for item in list(domain_ontology.values())[0]:
            for anno in item['annotations']:
                slot = anno.strip()
                if slot in anno2slot[domain]:
                    if anno2slot[domain][slot] in [None, False]:
                        continue
                    else:
                        slot = anno2slot[domain][slot]
                ontology['domains'][domain]['slots'][slot] = {
                    'description': descriptions[domain][slot],
                    'is_categorical': False,
                    'possible_values': [],
                }
                ontology['state'][domain][slot] = ''
    # add missing slots to the ontology
    for domain, slot in [('movies', 'price.streaming'), ('restaurant-search', 'phone')]:
        ontology['domains'][domain]['slots'][slot] = {
            'description': descriptions[domain][slot],
            'is_categorical': False,
            'possible_values': [],
        }
        ontology['state'][domain][slot] = ''

    dataset = 'tm2'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split:[] for split in splits}
    for domain in domains:
        data = json.load(open(os.path.join(original_data_dir, f"TM-2-2020/data/{domain}.json")))
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

        for dial_id, d in tqdm(enumerate(data), desc='processing taskmaster-{}'.format(domain)):
            # delete empty dialogs and invalid dialogs
            if len(d['utterances']) == 0:
                continue
            if len(set([t['speaker'] for t in d['utterances']])) == 1:
                continue
            data_split = dial_id2split[dial_id]
            dialogue_id = f'{dataset}-{data_split}-{len(dialogues_by_split[data_split])}'
            cur_domains = [domain]
            dialogue = {
                'dataset': dataset,
                'data_split': data_split,
                'dialogue_id': dialogue_id,
                'original_id': d["conversation_id"],
                'domains': cur_domains,
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
                        slot = item['name'].split('.', 1)[-1].strip()
                        if slot in anno2slot[domain]:
                            if anno2slot[domain][slot] is None:
                                # skip
                                continue
                            elif anno2slot[domain][slot] is False:
                                # binary dialog act
                                turn['dialogue_acts']['binary'].append({
                                    'intent': intent,
                                    'domain': domain,
                                    'slot': slot,
                                })
                                continue
                            else:
                                slot = anno2slot[domain][slot]
                        assert slot in ontology['domains'][domain]['slots'], print(domain, [slot])
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
                    prev_state[domain][slot] = value
                
                if speaker == 'user':
                    turn['state'] = copy.deepcopy(prev_state)

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
