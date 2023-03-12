# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import logging


class DatasetInterfacer(object):
    _domain_map_trippy_to_udf = {}
    _slot_map_trippy_to_udf = {}
    _generic_referral = {}

    def __init__(self):
        pass

    def map_trippy_to_udf(self, domain, slot):
        d = self._domain_map_trippy_to_udf.get(domain, domain)
        s = slot
        if d in self._slot_map_trippy_to_udf:
            s = self._slot_map_trippy_to_udf[d].get(slot, slot)
        return d, s

    def get_generic_referral(self, domain, slot):
        d, s = self.map_trippy_to_udf(domain, slot)
        ref = "the %s %s" % (d, s)
        if d in self._generic_referral:
            ref = self._generic_referral[d].get(s, s)
        return ref

    def normalize_values(self, text):
        return text

    def normalize_text(self, text):
        return text

    def normalize_prediction(self, domain, slot, value, predictions=None, config=None):
        return value


class MultiwozInterfacer(DatasetInterfacer):
    _slot_map_trippy_to_udf = {
        'hotel': {
            'pricerange': 'price range',
            'book_stay': 'book stay',
            'book_day': 'book day',
            'book_people': 'book people',
            'addr': 'address',
            'post': 'postcode',
            'price': 'price range',
            'people': 'book people'
        },
        'restaurant': {
            'pricerange': 'price range',
            'book_time': 'book time',
            'book_day': 'book day',
            'book_people': 'book people',
            'addr': 'address',
            'post': 'postcode',
            'price': 'price range',
            'people': 'book people'
        },
        'taxi': {
            'arriveBy': 'arrive by',
            'leaveAt': 'leave at',
            'arrive': 'arrive by',
            'leave': 'leave at',
            'car': 'type',
            'car type': 'type',
            'depart': 'departure',
            'dest': 'destination'
        },
        'train': {
            'arriveBy': 'arrive by',
            'leaveAt': 'leave at',
            'book_people': 'book people',
            'arrive': 'arrive by',
            'leave': 'leave at',
            'depart': 'departure',
            'dest': 'destination',
            'id': 'train id',
            'people': 'book people',
            'time': 'duration',
            'ticket': 'price',
            'trainid': 'train id'
        },
        'attraction': {
            'post': 'postcode',
            'addr': 'address',
            'fee': 'entrance fee',
            'price': 'entrance fee'
        },
        'general': {},
        'hospital': {
            'post': 'postcode',
            'addr': 'address'
        },
        'police': {
            'post': 'postcode',
            'addr': 'address'
        }
    }

    _generic_referral = {
        'hotel': {
            'name': 'the hotel',
            'area': 'same area as the hotel',
            'price range': 'in the same price range as the hotel'
        },
        'restaurant': {
            'name': 'the restaurant',
            'area': 'same area as the restaurant',
            'price range': 'in the same price range as the restaurant'
        },
        'attraction': {
            'name': 'the attraction',
            'area': 'same area as the attraction'
        }
    }
    
    def normalize_values(self, text):
        text = text.lower()
        text_to_num = {"zero": "0", "one": "1", "me": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7"}
        text = re.sub("\s*(\W)\s*", r"\1" , text) # Re-attach special characters
        text = re.sub("s'([^s])", r"s' \1", text) # Add space after plural genitive apostrophe
        if text in text_to_num:
            text = text_to_num[text]
        return text

    def normalize_text(self, text):
        norm_text = text.lower()
        #norm_text = re.sub("n't", " not", norm_text) # Does not make much of a difference
        norm_text = ' '.join([tok for tok in map(str.strip, re.split("(\W+)", norm_text)) if len(tok) > 0])
        return norm_text

    def normalize_prediction(self, domain, slot, value, predictions=None, class_predictions=None, config=None):
        v = value
        if domain == 'hotel' and slot == 'type':
            # Map Boolean predictions to regular predictions.
            v = "hotel" if value == "yes" else value
            v = "guesthouse" if value == "no" else value
            # HOTFIX: Avoid overprediction of hotel type caused by ambiguous rule based user simulator NLG.
            if predictions['hotel-name'] != 'none':
                v = 'none'
            if config.dst_class_types[class_predictions['hotel-none']] == 'request':
                v = 'none'
        return v


DATASET_INTERFACERS = {
    'multiwoz21': MultiwozInterfacer()
}


def create_dataset_interfacer(dataset_name="multiwoz21"):
    if dataset_name in DATASET_INTERFACERS:
        return DATASET_INTERFACERS[dataset_name]
    else:
        logging.warn("You attempt to create a dataset interfacer for an unknown dataset '%s'. Creating generic dataset interfacer." % (dataset_name))
        return DatasetInterfacer()


