# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
# Code adapted from the TRADE preprocessing code (https://github.com/jasonwu0731/trade-dst)
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
"""MultiWOZ2.1/3 data processing utilities"""

import re
import os

from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from convlab2.dst.rule.multiwoz import normalize_value

# ACTIVE_DOMAINS = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']
ACTIVE_DOMAINS = ['attraction', 'hotel', 'restaurant', 'taxi', 'train', 'hospital', 'police']
def set_util_domains(domains):
    global ACTIVE_DOMAINS
    ACTIVE_DOMAINS = [d for d in domains if d in ACTIVE_DOMAINS]

MAPPING_PATH = os.path.abspath(__file__).replace('utils.py', 'mapping.pair')
# Read replacement pairs from the mapping.pair file
REPLACEMENTS = []
for line in open(MAPPING_PATH).readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

# Extract belief state from mturk annotations
def build_dialoguestate(metadata, get_domains=False):
    domains_list = [dom for dom in ACTIVE_DOMAINS if dom in metadata]
    dialogue_state, domains = [], []
    for domain in domains_list:
        active = False
        # Extract booking information
        booking = []
        for slot in sorted(metadata[domain]['book'].keys()):
            if slot != 'booked':
                if metadata[domain]['book'][slot] == 'not mentioned':
                    continue
                if metadata[domain]['book'][slot] != '':
                    val = ['%s-book %s' % (domain, slot.strip().lower()), clean_text(metadata[domain]['book'][slot])]
                    dialogue_state.append(val)
                    active = True

        for slot in metadata[domain]['semi']:
            if metadata[domain]['semi'][slot] == 'not mentioned':
                continue
            elif metadata[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", 'don not care',
                                                    'do not care', 'does not care']:
                dialogue_state.append(['%s-%s' % (domain, slot.strip().lower()), 'do not care'])
                active = True
            elif metadata[domain]['semi'][slot]:
                dialogue_state.append(['%s-%s' % (domain, slot.strip().lower()), clean_text(metadata[domain]['semi'][slot])])
                active = True

        if active:
            domains.append(domain)

    if get_domains:
        return domains
    return clean_dialoguestate(dialogue_state)


PRICERANGE = ['do not care', 'cheap', 'moderate', 'expensive']
BOOLEAN = ['do not care', 'yes', 'no']
DAYS = ['do not care', 'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saterday', 'sunday']
QUANTITIES = ['do not care', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10 or more']
TIME = [[(i, j) for i in range(24)] for j in range(0, 60, 5)]
TIME = ['do not care'] + ['%02i:%02i' % t for l in TIME for t in l]

VALUE_MAP = {'guesthouse': 'guest house', 'belfry': 'belfray', '-': ' ', '&': 'and', 'b and b': 'bed and breakfast',
            'cityroomz': 'city roomz', '  ': ' ', 'acorn house': 'acorn guest house', 'marriot': 'marriott',
            'worth house': 'the worth house', 'alesbray lodge guest house': 'aylesbray lodge',
            'huntingdon hotel': 'huntingdon marriott hotel', 'huntingd': 'huntingdon marriott hotel',
            'jamaicanchinese': 'chinese', 'barbequemodern european': 'modern european',
            'north americanindian': 'north american', 'caribbeanindian': 'indian', 'sheeps': "sheep's"}

def map_values(value):
    for old, new in VALUE_MAP.items():
        value = value.replace(old, new)
    return value

def clean_dialoguestate(states, is_acts=False):
    # path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    # path = os.path.join(path, 'data/multiwoz/value_dict.json')
    # value_dict = json.load(open(path))
    clean_state = []
    for slot, value in states:
        if 'pricerange' in slot:
            d, s = slot.split('-', 1)
            s = 'price range'
            slot = f'{d}-{s}'
            if value in PRICERANGE:
                clean_state.append([slot, value])
            elif True in [v in value for v in PRICERANGE]:
                value = [v for v in PRICERANGE if v in value][0]
                clean_state.append([slot, value])
            elif value == '?' and is_acts:
                clean_state.append([slot, value])
            else:
                continue
        elif 'parking' in slot or 'internet' in slot:
            if value in BOOLEAN:
                clean_state.append([slot, value])
            if value == 'free':
                value = 'yes'
                clean_state.append([slot, value])
            elif True in [v in value for v in BOOLEAN]:
                value = [v for v in BOOLEAN if v in value][0]
                clean_state.append([slot, value])
            elif value == '?' and is_acts:
                clean_state.append([slot, value])
            else:
                continue
        elif 'day' in slot:
            if value in DAYS:
                clean_state.append([slot, value])
            elif True in [v in value for v in DAYS]:
                value = [v for v in DAYS if v in value][0]
                clean_state.append([slot, value])
            else:
                continue
        elif 'people' in slot or 'duration' in slot or 'stay' in slot:
            if value in QUANTITIES:
                clean_state.append([slot, value])
            elif True in [v in value for v in QUANTITIES]:
                value = [v for v in QUANTITIES if v in value][0]
                clean_state.append([slot, value])
            elif value == '?' and is_acts:
                clean_state.append([slot, value])
            else:
                try:
                    value = int(value)
                    if value >= 10:
                        value = '10 or more'
                        clean_state.append([slot, value])
                    else:
                        continue
                except:
                    continue
        elif 'time' in slot or 'leaveat' in slot or 'arriveby' in slot:
            if 'leaveat' in slot:
                d, s = slot.split('-', 1)
                s = 'leave at'
                slot = f'{d}-{s}'
            if 'arriveby' in slot:
                d, s = slot.split('-', 1)
                s = 'arrive by'
                slot = f'{d}-{s}'
            if value in TIME:
                if value == 'do not care':
                    clean_state.append([slot, value])
                else:
                    h, m = value.split(':')
                    if int(m) % 5 == 0:
                        clean_state.append([slot, value])
                    else:
                        m = round(int(m) / 5) * 5
                        h = int(h)
                        if m == 60:
                            m = 0
                            h += 1
                        if h >= 24:
                            h -= 24
                        value = '%02i:%02i' % (h, m)
                        clean_state.append([slot, value])
            elif True in [v in value for v in TIME]:
                value = [v for v in TIME if v in value][0]
                h, m = value.split(':')
                if int(m) % 5 == 0:
                    clean_state.append([slot, value])
                else:
                    m = round(int(m) / 5) * 5
                    h = int(h)
                    if m == 60:
                        m = 0
                        h += 1
                    if h >= 24:
                        h -= 24
                    value = '%02i:%02i' % (h, m)
                    clean_state.append([slot, value])
            elif value == '?' and is_acts:
                clean_state.append([slot, value])
            else:
                continue
        elif 'stars' in slot:
            if len(value) == 1 or value == 'do not care':
                clean_state.append([slot, value])
            elif value == '?' and is_acts:
                clean_state.append([slot, value])
            elif len(value) > 1:
                try:
                    value = int(value[0])
                    value = str(value)
                    clean_state.append([slot, value])
                except:
                    continue
        elif 'area' in slot:
            if '|' in value:
                value = value.split('|', 1)[0]
            clean_state.append([slot, value])
        else:
            if '|' in value:
                value = value.split('|', 1)[0]
                value = map_values(value)
                # d, s = slot.split('-', 1)
                # value = normalize_value(value_dict, d, s, value)
            clean_state.append([slot, value])
    
    return clean_state


# Module to process a dialogue and check its validity
def process_dialogue(dialogue, max_utt_len=128):
    if len(dialogue['log']) % 2 != 0:
        return None

    # Extract user and system utterances
    usr_utts, sys_utts = [], []
    avg_len = sum(len(utt['text'].split(' ')) for utt in dialogue['log'])
    avg_len = avg_len / len(dialogue['log'])
    if avg_len > max_utt_len:
        return None

    # If the first term is a system turn then ignore dialogue
    if dialogue['log'][0]['metadata']:
        return None

    usr, sys = None, None
    for turn in dialogue['log']:
        if not is_ascii(turn['text']):
            return None

        if not usr or not sys:
            if len(turn['metadata']) == 0:
                usr = turn
            else:
                sys = turn
        
        if usr and sys:
            states = build_dialoguestate(sys['metadata'], get_domains = False)
            sys['dialogue_states'] = states

            usr_utts.append(usr)
            sys_utts.append(sys)
            usr, sys = None, None

    dial_clean = dict()
    dial_clean['usr_log'] = usr_utts
    dial_clean['sys_log'] = sys_utts
    return dial_clean


# Get new domains
def get_act_domains(prev, crnt):
    diff = {}
    if not prev or not crnt:
        return diff

    for ((prev_dom, prev_val), (crnt_dom, crnt_val)) in zip(prev.items(), crnt.items()):
        assert prev_dom == crnt_dom
        if prev_val != crnt_val:
            diff[crnt_dom] = crnt_val
    return diff


# Get current domains
def get_domains(dial_log, turn_id, prev_domain):
    if turn_id == 1:
        active = build_dialoguestate(dial_log[turn_id]['metadata'], get_domains=True)
        acts = format_acts(dial_log[turn_id].get('dialog_act', {})) if not active else []
        acts = [domain for intent, domain, slot, value in acts if domain not in ['', 'general']]
        active += acts
        crnt = active[0] if active else ''
    else:
        active = get_act_domains(dial_log[turn_id - 2]['metadata'], dial_log[turn_id]['metadata'])
        active = list(active.keys())
        acts = format_acts(dial_log[turn_id].get('dialog_act', {})) if not active else []
        acts = [domain for intent, domain, slot, value in acts if domain not in ['', 'general']]
        active += acts
        crnt = [prev_domain] if not active else active
        crnt = crnt[0]

    return crnt


# Function to extract dialogue info from data
def extract_dialogue(dialogue, max_utt_len=50):
    dialogue = process_dialogue(dialogue, max_utt_len)
    if not dialogue:
        return None

    usr_utts = [turn['text'] for turn in dialogue['usr_log']]
    sys_utts = [turn['text'] for turn in dialogue['sys_log']]
    # sys_acts = [format_acts(turn['dialog_act']) if 'dialog_act' in turn else [] for turn in dialogue['sys_log']]
    usr_acts = [format_acts(turn['dialog_act']) if 'dialog_act' in turn else [] for turn in dialogue['usr_log']]
    dialogue_states = [turn['dialogue_states'] for turn in dialogue['sys_log']]
    domains = [turn['domain'] for turn in dialogue['usr_log']]

    # dial = [{'usr': u,'sys': s, 'usr_a': ua, 'sys_a': a, 'domain': d, 'ds': v}
    #         for u, s, ua, a, d, v in zip(usr_utts, sys_utts, usr_acts, sys_acts, domains, dialogue_states)]
    dial = [{'usr': u,'sys': s, 'usr_a': ua, 'domain': d, 'ds': v}
            for u, s, ua, d, v in zip(usr_utts, sys_utts, usr_acts, domains, dialogue_states)]    
    return dial


def format_acts(acts):
    new_acts = []
    for key, item in acts.items():
        domain, intent = key.split('-', 1)
        if domain.lower() in ACTIVE_DOMAINS + ['general']:
            state = []
            for slot, value in item:
                slot = str(REF_SYS_DA[domain].get(slot, slot)).lower() if domain in REF_SYS_DA else slot
                value = clean_text(value)
                slot = slot.replace('_', ' ').replace('ref', 'reference')
                state.append([f'{domain.lower()}-{slot}', value])
            state = clean_dialoguestate(state, is_acts=True)
            if domain == 'general':
                if intent in ['thank', 'bye']:
                    state = [['general-none', 'none']]
                else:
                    state = []
            for slot, value in state:
                if slot not in ['train-people']:
                    slot = slot.split('-', 1)[-1]
                    new_acts.append([intent.lower(), domain.lower(), slot, value])
    
    return new_acts
                

# Fix act labels
def fix_delexicalisation(turn):
    if 'dialog_act' in turn:
        for dom, act in turn['dialog_act'].items():
            if 'Attraction' in dom:
                if 'restaurant_' in turn['text']:
                    turn['text'] = turn['text'].replace("restaurant", "attraction")
                if 'hotel_' in turn['text']:
                    turn['text'] = turn['text'].replace("hotel", "attraction")
            if 'Hotel' in dom:
                if 'attraction_' in turn['text']:
                    turn['text'] = turn['text'].replace("attraction", "hotel")
                if 'restaurant_' in turn['text']:
                    turn['text'] = turn['text'].replace("restaurant", "hotel")
            if 'Restaurant' in dom:
                if 'attraction_' in turn['text']:
                    turn['text'] = turn['text'].replace("attraction", "restaurant")
                if 'hotel_' in turn['text']:
                    turn['text'] = turn['text'].replace("hotel", "restaurant")

    return turn


# Check if a character is an ascii character
def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# Insert white space
def separate_token(token, text):
    sidx = 0
    while True:
        # Find next instance of token
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        # If the token is already seperated continue to next
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        # Create white space separation around token
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def clean_text(text):
    # Replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text.strip().lower())

    # Replace b&v or 'b and b' with 'bed and breakfast'
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # Fix apostrophies
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # Correct punctuation
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # Replace special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)

    # Insert white space around special tokens:
    for token in ['?', '.', ',', '!']:
        text = separate_token(token, text)

    # insert white space for 's
    text = separate_token('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    # Perform pair replacements listed in the mapping.pair file
    for fromx, tox in REPLACEMENTS:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # Remove multiple spaces
    text = re.sub(' +', ' ', text)

    # Concatenate numbers eg '1 3' -> '13'
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text
