import re
import json
import random
from collections import Counter

import torch
import numpy as np

from convlab.util import *

DEFAULTS = {
    'ref': '00000000',
    'bookpeople': '2',
    'bookday': 'Tuesday'
}

SLOT_MAPS = {
    'entrancefee': 'entrance fee',
    'trainid': 'trainID',
    'leaveat': 'leaveAt',
    'arriveby': 'arriveBy'
}

TAXI_SLOT_MAPS = {
    'type': 'taxi_types',
    'color': 'taxi_colors',
    'phone': 'taxi_phone'
    
}

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def find_substrings(input_string, start_tag, end_tag):
    pattern = fr'{re.escape(start_tag)}(.*?){re.escape(end_tag)}'
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches

dataset_name = 'multiwoz21'
database = load_database(dataset_name)
 
def lexcalise(full_sequence, database):
    # r = turn['response']
    # response = full_sequence[full_sequence.find('<|response|>')+len('<|response|>'):full_sequence.find('<|endofresponse|>')]
    response = find_substrings(full_sequence, '<|response|>', '<|endofresponse|>')
    if not response:
        return ''
    else:
        response = response[-1]
    # pattern = r'\[([^\]]+)\]'
    # slots_to_fill = re.findall(pattern, response)
    # if len(slots_to_fill) == 0:
    #     return response
    slots_to_fill = find_substrings(full_sequence, '[', ']')
    if not slots_to_fill:
        return response

    # g = turn['generated']

    # if ('<|action|>' not in full_sequence) or ('<|endofaction|>' not in full_sequence):
    #     raise ValueError(f'action not available')
    
    # actions = full_sequence[full_sequence.find('<|action|>')+len('<|action|>'):full_sequence.find('<|endofaction|>')].strip()
    actions = find_substrings(full_sequence, '<|action|>', '<|endofaction|>')
    if not actions:
        action_triplets = []
    else:
        action_triplets = [s.strip().split(' ') for s in actions[-1].strip().split(',')]
    
    action_domains = []
    active_domain = None
    informable_domains = ['restaurant', 'attraction', 'hotel', 'train', 'taxi', 'hospital'] #, 'booking', 'general']
    for das in action_triplets:
        d = das[0]
        a = das[1]
        if a == 'inform' and d in informable_domains:
            action_domains.append(d)
    if len(action_domains) > 0:
        act_domain_counts = Counter(action_domains)
        active_domain, count = act_domain_counts.most_common(1)[0]
    else:
        for a in informable_domains:
            if a in action_domains:
                active_domain = a
                break

    # if ('<|belief|>' not in g) or ('<|endofbelief|>' not in g):
    #     raise ValueError(f'belief state not available in turn {str(i)} of dialogue {uid}: {g}')
    # bs = full_sequence[full_sequence.find('<|belief|>')+len('<|belief|>'):full_sequence.find('<|endofbelief|>')].strip()
    bs = find_substrings(full_sequence, '<|belief|>', '<|endofbelief|>')
    if not bs:
        bs_tuples = []
    else:
        bs_tuples = [s.strip().split(' ') for s in bs[-1].strip().split(',')]
    constraints = {}
    dialog_state = {}
    for dsv in bs_tuples:
        if len(dsv) > 1:
            domain = dsv[0]
            slot = dsv[1]
            if slot == 'book':
                value = ' '.join(dsv[3:])
                slot = dsv[1] + dsv[2]
            else:
                value = ' '.join(dsv[2:])
            value = normalize_state_slot_value(slot, value)
            dialog_state[slot] = value
            if domain not in constraints:
                constraints[domain] = []
            constraints[domain].append([slot, value])

    try:
        domain_entities = {}
        for d in set(action_domains):
            entities = database.query(d, [], topk=1000)
            domain_entities[d] = entities
        for d in constraints:   # from bs
            entities = []
            if d is not None:
                entities = database.query(d, constraints[d], topk=1000)
            domain_entities[d] = entities

    except:
        raise ValueError(f'Error in database query')
    
    # selected_entities = []
    # for d in domain_entities:
    #     if len(domain_entities[d]) > 0:
    #         selected_entity = domain_entities[d][0]
    #         selected_entities.append([d, selected_entity])

    lexicalised_response = response
    filled_slot_count = {}
    for s in slots_to_fill:
        value = 'unknown'
        if s in dialog_state:
            value = dialog_state[s]
        elif s in DEFAULTS:
            value = DEFAULTS[s]
        elif s == 'choice':
            value = str(len(entities))
        else:
            for d, all_entities in domain_entities.items():
                if s not in filled_slot_count:
                    try:
                        entity = all_entities[0]
                    except:
                        entity = {}
                else:
                    try:
                        entity = all_entities[filled_slot_count[s]]
                    except:
                        entity = {}

                if s in entity:
                    value = entity[s]
                    break
                elif s in SLOT_MAPS and SLOT_MAPS[s] in entity:
                    value = entity[SLOT_MAPS[s]]
                    break                        
                else:
                    if d == 'taxi':
                        if s in TAXI_SLOT_MAPS:
                            value =  entity[TAXI_SLOT_MAPS[s]]
                            break
                    elif d == 'hospital':
                        if s == 'name':
                            value = "Addenbrooke's"
                            break

        if isinstance(value, dict):
            value = str(value)

        if s not in filled_slot_count:
            filled_slot_count[s] = 0
        filled_slot_count[s] += 1

        lexicalised_response = lexicalised_response.replace(f'[{s}]', value, 1)
    
    # slots_yet_to_fill = re.findall(pattern, lexicalised_response)
    return lexicalised_response


# https://github.com/Tomiinek/MultiWOZ_Evaluation
def normalize_state_slot_value(slot_name, value):
    """ Normalize slot value:
        1) replace too distant venue names with canonical values
        2) replace too distant food types with canonical values
        3) parse time strings to the HH:MM format
        4) resolve inconsistency between the database entries and parking and internet slots
    """
    
    def type_to_canonical(type_string): 
        if type_string == "swimming pool":
            return "swimmingpool" 
        elif type_string == "mutliple sports":
            return "multiple sports"
        elif type_string == "night club":
            return "nightclub"
        elif type_string == "guest house":
            return "guesthouse"
        return type_string

    def name_to_canonical(name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        name = name.replace(" '", "'")
        
        name = name.replace("bed and breakfast","b and b")
        
        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
            elif name == "restaurant 2 two":
                return "restaurant two two"
            elif name == "restaurant two 2":
                return "restaurant two two"
            elif name == "restaurant 2 2":
                return "restaurant two two"
            elif name == "restaurant 1 7" or name == "restaurant 17":
                return "restaurant one seven"
            # new
            elif name == "copper kettle":
                return "the copper kettle"

        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"
            elif name == "a and b guest house":
                return "a and b guesthouse"
            elif name == "acorn guesthouse":
                return "acorn guest house"
            elif name == "cambridge belfry":
                return "the cambridge belfry"
            

        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"
            elif name == 'pizza hut fenditton':
                return 'pizza hut fen ditton'
            elif name == 'cafe jello gallery':
                return 'cafe jello museum'
        return name

    def time_to_canonical(time):
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()    

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1] 
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'
        
        if len(time) == 0:
            return "00:00"
            
        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]
            
        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"
        
        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def food_to_canonical(food):
        """ Converts food name to caninical form used in database. """

        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "sea food": return "seafood"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"
        if food == "europeon": return "european"

        return food

    if slot_name in ["name", "destination", "departure"]:
        return name_to_canonical(value)
    elif slot_name == "type":
        return type_to_canonical(value)
    elif slot_name == "food":
        return food_to_canonical(value)
    elif slot_name in ["arrive", "leave", "arriveby", "leaveat", "time"]:
        return time_to_canonical(value)
    elif slot_name in ["parking", "internet"]:
        return "yes" if value == "free" else value
    else:
        return value