
def normalize_slot_name(slot_name):
    """ Map a slot name to the new unified ontology. """

    slot_name = slot_name.lower()
    slot_name_mapping = {
     'ADDRESS'   : ['address', 'attraction_address', 'hospital_address', 'hotel_address', 'police_address', 'restaurant_address', 'value_address'],
     'AREA'      : ['area', 'value_area', 'attraction_area', 'restaurant_area', 'hotel_area'],
     'TIME'      : ['booktime', 'value_time', 'time', 'duration', 'value_duration', 'train_duration', 'arriveby', 'taxi_arriveby', 'value_arrive', 'arrive by', 'train_arriveby', 'leaveat', 'value_leave', 'leave at', 'train_leaveat', 'train_leave', 'train_arrive', 'taxi_leaveat'],
     'DAY'       : ['day', 'value_day', 'bookday', 'train_day'],
     'PLACE'     : ['destination', 'value_destination', 'departure', 'value_departure', 'value_place', 'train_departure', 'train_destination', 'taxi_destination', 'taxi_departure'],
     'FOOD'      : ['food', 'value_food', 'restaurant_food'],
     'NAME'      : ['name', 'attraction_name', 'hospital_name', 'hotel_name', 'police_name', 'restaurant_name', 'value_name'],
     'PHONE'     : ['phone', 'attraction_phone', 'hospital_phone', 'hotel_phone', 'police_phone', 'restaurant_phone', 'taxi_phone', 'value_phone'],
     'POST'      : ['postcode', 'attraction_postcode', 'hospital_postcode', 'hotel_postcode', 'restaurant_postcode', 'value_postcode', 'police_postcode'],
     'PRICE'     : ['price', 'value_price', 'entrancefee', 'entrance fee', 'train_price', 'attraction_entrancefee', 'pricerange', 'value_pricerange', 'price range', 'restaurant_pricerange', 'hotel_pricerange', 'attraction_pricerange', 'attraction_price'],
     'REFERENCE' : ['ref', 'attraction_reference', 'hotel_reference', 'restaurant_reference', 'train_reference', 'value_reference', 'reference'],  
     'COUNT'     : ['stars', 'value_stars', 'hotel_stars', 'bookstay', 'value_stay', 'stay', 'bookpeople', 'value_people', 'people', 'choice', 'value_choice', 'value_count', 'attraction_choice', 'hotel_choice', 'restaurant_choice', 'train_choice'],
     'TYPE'      : ['type', 'taxi_type', 'taxi_car', 'value_type', 'value_car', 'car', 'restaurant_type', 'hotel_type', 'attraction_type'],
     'TRAINID'   : ['trainid', 'train_id', 'value_id', 'id', 'train', 'train_trainid'],
     'INTERNET'  : ['internet', 'hotel_internet'],
     'PARKING'   : ['parking', 'hotel_parking'],
     'ID'        : ['hospital_id', 'attraction_id', 'restaurant_id'],
     'DEPARTMENT': ['value_department', 'department', 'hospital_department'],
     'OPEN'      : ['openhours']
    }
    reverse_slot_name_mapping = {s : k for k, v in slot_name_mapping.items() for s in v}  
    if slot_name not in reverse_slot_name_mapping:
        # print(f"Unknown slot name: {slot_name}. Please use another slot names or customize the slot mapping!")
        return ''
    return reverse_slot_name_mapping[slot_name]

# import re
# from functools import partial
# from utils import normalize_slot_name
# slot_name_re = re.compile(r'\[([\w\s\d]+)\](es|s|-s|-es|)')
# slot_name_normalizer = partial(slot_name_re.sub, lambda x: normalize_slot_name(x.group(1)))

import numpy as np
import os, random, json, re
import spacy
from copy import deepcopy
from collections import Counter

from convlab.e2e.soloist.multiwoz.db_ops import MultiWozDB
from convlab.e2e.soloist.multiwoz.config import global_config as cfg

DEFAULT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 'not','of','on','or','so','the','there','was','were','whatever','whether','would']

class MultiWozReader(object):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(DEFAULT_DIRECTORY, cfg.dbs)

    def restore(self, resp, domain, constraint_dict):
        restored = resp
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')


        mat_ents = self.db.get_match_num(constraint_dict, True)

        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace('[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace('[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace('[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace('[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        # restored.replace('[value_car]', 'BMW')


        ent = mat_ents.get(domain[-1], [])
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    slot = t[7:-1]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)

            # handle normal 1 entity case
            ent = ent[0]
            for t in restored.split():
                if '[value' in t:
                    slot = t[7:-1]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        # else:
                        #     print(restored, domain)


        restored = restored.replace('[value_phone]', '01223462354')
        restored = restored.replace('[value_postcode]', 'CB12DP')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')

        for t in restored.split():
            if '[value' in t:
                restored = restored.replace(t, 'UNKNOWN')

        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)

        return restored
