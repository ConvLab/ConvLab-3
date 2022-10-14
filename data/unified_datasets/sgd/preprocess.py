from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
from pprint import pprint
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
from shutil import rmtree
import re

digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
}

match = {
    '0': 0,
    '1': 0,
    '>1': 0,
}


def pharse_in_sen(phrase, sen):
    '''
    match value in the sentence
    :param phrase: str
    :param sen: str
    :return: start, end if matched, else None, None
    '''
    assert isinstance(phrase, str)
    pw = '(^|[\s,\.:\?!-])(?P<v>{})([\s,\.:\?!-]|$)'
    pn = '(^|[\s\?!-]|\D[,\.:])(?P<v>{})($|[\s\?!-]|[,\.:]\D|[,\.:]$)'
    if phrase.isdigit():
        pattern = pn
    else:
        pattern = pw
    p = re.compile(pattern.format(re.escape(phrase)), re.I)
    m = re.search(p, sen)
    if m:
        num = len(re.findall(p, sen))
        if num > 1:
            match['>1'] += 1
        else:
            match['1'] += 1
        return m.span('v'), num
    if phrase.isdigit() and phrase in digit2word:
        phrase = digit2word[phrase]
        p = re.compile(pw.format(re.escape(phrase)), re.I)
        m = re.search(p, sen)
        if m:
            num = len(re.findall(p, sen))
            if num > 1:
                match['>1'] += 1
            else:
                match['1'] += 1
            return m.span('v'), num
    match['0'] += 1
    return (None, None), 0


def sys_intent():
    """from original data README.md"""
    return {
        "inform": {"description": "Inform the value for a slot to the user."},
        "request": {"description": "Request the value of a slot from the user."},
        "confirm": {"description": "Confirm the value of a slot before making a transactional service call."},
        "offer": {"description": "Offer a certain value for a slot to the user."},
        "notify_success": {"description": "Inform the user that their request was successful."},
        "notify_failure": {"description": "Inform the user that their request failed."},
        "inform_count": {"description": "Inform the number of items found that satisfy the user's request."},
        "offer_intent": {"description": "Offer a new intent to the user."},
        "req_more": {"description": "Asking the user if they need anything else."},
        "goodbye": {"description": "End the dialogue."},
    }


def usr_intent():
    """from original data README.md"""
    return {
        "inform_intent": {"description": "Express the desire to perform a certain task to the system."},
        "negate_intent": {"description": "Negate the intent which has been offered by the system."},
        "affirm_intent": {"description": "Agree to the intent which has been offered by the system."},
        "inform": {"description": "Inform the value of a slot to the system."},
        "request": {"description": "Request the value of a slot from the system."},
        "affirm": {"description": "Agree to the system's proposition. "},
        "negate": {"description": "Deny the system's proposal."},
        "select": {"description": "Select a result being offered by the system."},
        "request_alts": {"description": "Ask for more results besides the ones offered by the system."},
        "thank_you": {"description": "Thank the system."},
        "goodbye": {"description": "End the dialogue."},
    }


def get_intent():
    """merge sys & usr intent"""
    return {
        "inform": {"description": "Inform the value for a slot."},
        "request": {"description": "Request the value of a slot."},
        "confirm": {"description": "Confirm the value of a slot before making a transactional service call."},
        "offer": {"description": "Offer a certain value for a slot to the user."},
        "notify_success": {"description": "Inform the user that their request was successful."},
        "notify_failure": {"description": "Inform the user that their request failed."},
        "inform_count": {"description": "Inform the number of items found that satisfy the user's request."},
        "offer_intent": {"description": "Offer a new intent to the user."},
        "req_more": {"description": "Asking the user if they need anything else."},
        "goodbye": {"description": "End the dialogue."},
        "inform_intent": {"description": "Express the desire to perform a certain task to the system."},
        "negate_intent": {"description": "Negate the intent which has been offered by the system."},
        "affirm_intent": {"description": "Agree to the intent which has been offered by the system."},
        "affirm": {"description": "Agree to the system's proposition. "},
        "negate": {"description": "Deny the system's proposal."},
        "select": {"description": "Select a result being offered by the system."},
        "request_alts": {"description": "Ask for more results besides the ones offered by the system."},
        "thank_you": {"description": "Thank the system."},
    }


def preprocess():
    original_data_dir = 'dstc8-schema-guided-dialogue-master'
    new_data_dir = 'data'

    if not os.path.exists(original_data_dir):
        original_data_zip = 'dstc8-schema-guided-dialogue-master.zip'
        if not os.path.exists(original_data_zip):
            raise FileNotFoundError(f'cannot find original data {original_data_zip} in sgd/, should manually download dstc8-schema-guided-dialogue-master.zip from https://github.com/google-research-datasets/dstc8-schema-guided-dialogue')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()
    
    os.makedirs(new_data_dir, exist_ok=True)

    dialogues = []
    ontology = {'domains': {},
                'intents': get_intent(),
                'state': {},
                'dialogue_acts': {
                    "categorical": {},
                    "non-categorical": {},
                    "binary": {}
                }}
    splits = ['train', 'validation', 'test']
    dataset_name = 'sgd'
    for data_split in splits:
        data_dir = os.path.join(original_data_dir, data_split if data_split != 'validation' else 'dev')
        # schema => ontology
        with open(os.path.join(data_dir, 'schema.json')) as f:
            data = json.load(f)
            for schema in data:
                domain = schema['service_name']
                ontology['domains'].setdefault(domain, {})
                ontology['domains'][domain]['description'] = schema['description']
                ontology['domains'][domain].setdefault('slots', {})
                ontology['domains'][domain]['active_intents'] = schema['intents']
                ontology['state'].setdefault(domain, {})
                for slot in schema['slots']:
                    ontology['domains'][domain]['slots'][slot['name']] = {
                        "description": slot['description'],
                        "is_categorical": slot['is_categorical'],
                        "possible_values": slot['possible_values']
                    }
                    ontology['state'][domain][slot['name']] = ''
                # add 'count' slot
                ontology['domains'][domain]['slots']['count'] = {
                    "description": "the number of items found that satisfy the user's request.",
                    "is_categorical": False,
                    "possible_values": []
                }

        # dialog
        cnt = 0
        for root, dirs, files in os.walk(data_dir):
            fs = sorted([x for x in files if 'dialogues' in x])
            for f in tqdm(fs, desc='processing schema-guided-{}'.format(data_split)):
                data = json.load(open(os.path.join(data_dir, f)))
                for d in data:
                    dialogue = {
                        "dataset": dataset_name,
                        "data_split": data_split,
                        "dialogue_id": f'{dataset_name}-{data_split}-{cnt}',
                        "original_id": d['dialogue_id'],
                        "domains": d['services'],
                        "goal": { # no goal
                            'description': '',
                            'inform': {},
                            'request': {}
                        },
                        "turns": []
                    }
                    cnt += 1
                    prev_state = {}
                    for domain in dialogue['domains']:
                        prev_state.setdefault(domain, deepcopy(ontology['state'][domain]))

                    for utt_idx, t in enumerate(d['turns']):
                        speaker = t['speaker'].lower()
                        turn = {
                            'speaker': speaker,
                            'utterance': t['utterance'],
                            'utt_idx': utt_idx,
                            'dialogue_acts': {
                                'binary': [],
                                'categorical': [],
                                'non-categorical': [],
                            },
                        }
                        for frame in t['frames']:
                            domain = frame['service']
                            for action in frame['actions']:
                                intent = action['act'].lower() # lowercase intent
                                assert intent in ontology['intents'], [intent]
                                slot = action['slot']
                                value_list = action['values']
                                if action['act'] in ['REQ_MORE', 'AFFIRM', 'NEGATE', 'THANK_YOU', 'GOODBYE']:
                                    # Slot and values are always empty
                                    assert slot == "" and len(value_list) == 0
                                    turn['dialogue_acts']['binary'].append({
                                        "intent": intent,
                                        "domain": '',
                                        "slot": ''
                                    })
                                elif action['act'] in ['NOTIFY_SUCCESS', 'NOTIFY_FAILURE', 'REQUEST_ALTS', 'AFFIRM_INTENT', 'NEGATE_INTENT']:
                                    # Slot and values are always empty
                                    assert slot == "" and len(value_list) == 0
                                    turn['dialogue_acts']['binary'].append({
                                        "intent": intent,
                                        "domain": domain,
                                        "slot": ''
                                    })
                                elif action['act'] in ['OFFER_INTENT', 'INFORM_INTENT']:
                                    # slot containing the intent being offered.
                                    assert slot == 'intent' and len(value_list) == 1
                                    turn['dialogue_acts']['binary'].append({
                                        "intent": intent,
                                        "domain": domain,
                                        "slot": value_list[0]
                                    })
                                elif action['act'] in ['REQUEST'] and len(value_list) == 0:
                                    # always contains a slot, but values are optional.
                                    assert slot in ontology['domains'][domain]['slots'], f'{domain}-{slot}'
                                    turn['dialogue_acts']['binary'].append({
                                        "intent": intent,
                                        "domain": domain,
                                        "slot": slot
                                    })
                                elif action['act'] in ['SELECT'] and len(value_list) == 0:
                                    # (slot=='' and len(value_list) == 0) or (slot!='' and len(value_list) > 0)
                                    assert slot == '', f'{domain}-{slot}-{action}'
                                    turn['dialogue_acts']['binary'].append({
                                        "intent": intent,
                                        "domain": domain,
                                        "slot": slot
                                    })
                                elif action['act'] in ['INFORM_COUNT']:
                                    # always has "count" as the slot, and a single element in values for the number of results obtained by the system.
                                    assert slot == 'count' and len(value_list) == 1
                                    value = value_list[0]

                                    turn['dialogue_acts']['non-categorical'].append({
                                        "intent": intent,
                                        "domain": domain,
                                        "slot": slot,
                                        "value": value,
                                    })
                                    
                                    # find char span
                                    (start, end), num = pharse_in_sen(value, t['utterance'])
                                    assert num > 0, f'{value}-{t["utterance"]}' # {1:20086, 2:341, 3:19}
                                    assert value.lower() == t['utterance'][start:end].lower() \
                                                or digit2word[value].lower() == t['utterance'][start:end].lower()
                                    # first match is always the choice
                                    turn['dialogue_acts']['non-categorical'][-1].update({
                                        "value": t['utterance'][start:end], "start": start, "end": end
                                    })
                                else:
                                    # have slot & value
                                    if ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                        possible_values = [value.lower() for value in ontology['domains'][domain]['slots'][slot]['possible_values']]
                                        for value in value_list:
                                            if value.lower() not in possible_values and value != 'dontcare':
                                                ontology['domains'][domain]['slots'][slot]['possible_values'].append(value)
                                                print(f'add value to ontology\t{domain}-{slot}-{value}', possible_values)
                                            turn['dialogue_acts']['categorical'].append({
                                                "intent": intent,
                                                "domain": domain,
                                                "slot": slot,
                                                "value": value,
                                            })
                                    else:
                                        # span info in frame['slots']
                                        for value in value_list:
                                            for slot_info in frame['slots']:
                                                start = slot_info['start']
                                                end = slot_info['exclusive_end']
                                                if slot_info['slot'] == slot and t['utterance'][start:end].lower() == value.lower():
                                                    assert t['utterance'][start:end] == value, f'{action}-{slot_info}-{t["utterance"][start:end]}'
                                                    turn['dialogue_acts']['non-categorical'].append({
                                                        "intent": intent,
                                                        "domain": domain,
                                                        "slot": slot,
                                                        "value": value,
                                                        "start": start,
                                                        "end": end
                                                    })
                                                    break
                                            else:
                                                assert value == 'dontcare', f'{action}-{slot_info}'
                                                
                        if speaker == 'user':
                            state = deepcopy(prev_state)
                            active_intent = {}
                            requested_slots = {}
                            for frame in t['frames']:
                                domain = frame['service']
                                active_intent[domain] = frame['state']['active_intent']
                                requested_slots[domain] = frame['state']['requested_slots']
                                for slot in state[domain]:
                                    if slot in frame['state']['slot_values']:
                                        value_list = frame['state']['slot_values'][slot]
                                        state[domain][slot] = value_list[0]
                                        for value in value_list[1:]:
                                            state[domain][slot] += '|' + value
                                    else:
                                        state[domain][slot] = ''
                            prev_state = state
                            turn['state'] = state
                            turn['active_intent'] = active_intent
                            turn['requested_slots'] = requested_slots
                        else:
                            # service_call and service_results
                            turn['service_call'] = {}
                            turn['db_results'] = {}
                            for frame in t['frames']:
                                if 'service_call' not in frame:
                                    continue
                                domain = frame['service']
                                turn['service_call'][domain] = frame['service_call']
                                turn['db_results'][domain] = frame['service_results']

                        # add to dialogue_acts dictionary in the ontology
                        for da_type in turn['dialogue_acts']:
                            das = turn['dialogue_acts'][da_type]
                            for da in das:
                                ontology["dialogue_acts"][da_type].setdefault((da['intent'], da['domain'], da['slot']), {})
                                ontology["dialogue_acts"][da_type][(da['intent'], da['domain'], da['slot'])][speaker] = True
                        dialogue['turns'].append(turn)
                    dialogues.append(dialogue)
    
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str({'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent':da[0],'domain':da[1], 'slot':da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    # rmtree(original_data_dir)
    # rmtree(new_data_dir)
    return dialogues, ontology

if __name__ == '__main__':
    preprocess()