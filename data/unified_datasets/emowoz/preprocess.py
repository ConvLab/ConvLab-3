import copy
import re
from zipfile import ZipFile, ZIP_DEFLATED
from shutil import copy2, rmtree
import json
import os
from tqdm import tqdm
from collections import Counter
from pprint import pprint
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer

ontology = {
    "domains": {  # descriptions are adapted from multiwoz22, but is_categorical may be different
        "attraction": {
            "description": "find an attraction",
            "slots": {
                "area": {
                    "description": "area to search for attractions",
                    "is_categorical": True,
                    "possible_values": [
                        "centre",
                        "east",
                        "north",
                        "south",
                        "west"
                    ]
                },
                "name": {
                    "description": "name of the attraction",
                    "is_categorical": False,
                    "possible_values": []
                },
                "type": {
                    "description": "type of the attraction",
                    "is_categorical": True,
                    "possible_values": [
                        "architecture",
                        "boat",
                        "cinema",
                        "college",
                        "concerthall",
                        "entertainment",
                        "museum",
                        "multiple sports",
                        "nightclub",
                        "park",
                        "swimmingpool",
                        "theatre"
                    ]
                },
                "entrance fee": {
                    "description": "how much is the entrance fee",
                    "is_categorical": False,
                    "possible_values": []
                },
                "open hours": {
                    "description": "open hours of the attraction",
                    "is_categorical": False,
                    "possible_values": []
                },
                "address": {
                    "description": "address of the attraction",
                    "is_categorical": False,
                    "possible_values": []
                },
                "phone": {
                    "description": "phone number of the attraction",
                    "is_categorical": False,
                    "possible_values": []
                },
                "postcode": {
                    "description": "postcode of the attraction",
                    "is_categorical": False,
                    "possible_values": []
                },
                "choice": {
                    "description": "number of attractions that meet the requirement",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "hotel": {
            "description": "find and book a hotel",
            "slots": {
                "internet": {
                    "description": "whether the hotel has internet",
                    "is_categorical": True,
                    "possible_values": [
                        "free",
                        "no",
                        "yes"
                    ]
                },
                "parking": {
                    "description": "whether the hotel has parking",
                    "is_categorical": True,
                    "possible_values": [
                        "free",
                        "no",
                        "yes"
                    ]
                },
                "area": {
                    "description": "area or place of the hotel",
                    "is_categorical": True,
                    "possible_values": [
                        "centre",
                        "east",
                        "north",
                        "south",
                        "west"
                    ]
                },
                "stars": {
                    "description": "star rating of the hotel",
                    "is_categorical": True,
                    "possible_values": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5"
                    ]
                },
                "price range": {
                    "description": "price budget of the hotel",
                    "is_categorical": True,
                    "possible_values": [
                        "expensive",
                        "cheap",
                        "moderate"
                    ]
                },
                "type": {
                    "description": "what is the type of the hotel",
                    "is_categorical": False,
                    "possible_values": [
                        "guesthouse",
                        "hotel"
                    ]
                },
                "name": {
                    "description": "name of the hotel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book people": {
                    "description": "number of people for the hotel booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book stay": {
                    "description": "length of stay at the hotel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book day": {
                    "description": "day of the hotel booking",
                    "is_categorical": True,
                    "possible_values": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday"
                    ]
                },
                "phone": {
                    "description": "phone number of the hotel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "postcode": {
                    "description": "postcode of the hotel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "address": {
                    "description": "address of the hotel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "ref": {
                    "description": "reference number of the hotel booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "choice": {
                    "description": "number of hotels that meet the requirement",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "taxi": {
            "description": "rent taxi to travel",
            "slots": {
                "destination": {
                    "description": "destination of taxi",
                    "is_categorical": False,
                    "possible_values": []
                },
                "departure": {
                    "description": "departure location of taxi",
                    "is_categorical": False,
                    "possible_values": []
                },
                "leave at": {
                    "description": "leaving time of taxi",
                    "is_categorical": False,
                    "possible_values": []
                },
                "arrive by": {
                    "description": "arrival time of taxi",
                    "is_categorical": False,
                    "possible_values": []
                },
                "phone": {
                    "description": "phone number of the taxi",
                    "is_categorical": False,
                    "possible_values": []
                },
                "type": {
                    "description": "car type of the taxi",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "restaurant": {
            "description": "find and book a restaurant",
            "slots": {
                "price range": {
                    "description": "price budget for the restaurant",
                    "is_categorical": True,
                    "possible_values": [
                        "cheap",
                        "expensive",
                        "moderate"
                    ]
                },
                "area": {
                    "description": "area or place of the restaurant",
                    "is_categorical": True,
                    "possible_values": [
                        "centre",
                        "east",
                        "north",
                        "south",
                        "west"
                    ]
                },
                "food": {
                    "description": "the cuisine of the restaurant",
                    "is_categorical": False,
                    "possible_values": []
                },
                "name": {
                    "description": "name of the restaurant",
                    "is_categorical": False,
                    "possible_values": []
                },
                "address": {
                    "description": "address of the restaurant",
                    "is_categorical": False,
                    "possible_values": []
                },
                "postcode": {
                    "description": "postcode of the restaurant",
                    "is_categorical": False,
                    "possible_values": []
                },
                "phone": {
                    "description": "phone number of the restaurant",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book people": {
                    "description": "number of people for the restaurant booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book time": {
                    "description": "time of the restaurant booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book day": {
                    "description": "day of the restaurant booking",
                    "is_categorical": True,
                    "possible_values": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday"
                    ]
                },
                "ref": {
                    "description": "reference number of the restaurant booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "choice": {
                    "description": "number of restaurants that meet the requirement",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "train": {
            "description": "find a train to travel",
            "slots": {
                "destination": {
                    "description": "destination of the train",
                    "is_categorical": False,
                    "possible_values": []
                },
                "arrive by": {
                    "description": "arrival time of the train",
                    "is_categorical": False,
                    "possible_values": []
                },
                "departure": {
                    "description": "departure location of the train",
                    "is_categorical": False,
                    "possible_values": []
                },
                "leave at": {
                    "description": "leaving time for the train",
                    "is_categorical": False,
                    "possible_values": []
                },
                "duration": {
                    "description": "duration of the travel",
                    "is_categorical": False,
                    "possible_values": []
                },
                "book people": {
                    "description": "number of people booking for train",
                    "is_categorical": False,
                    "possible_values": []
                },
                "day": {
                    "description": "day of the train",
                    "is_categorical": True,
                    "possible_values": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday"
                    ]
                },
                "ref": {
                    "description": "reference number of the train booking",
                    "is_categorical": False,
                    "possible_values": []
                },
                "price": {
                    "description": "price of the train ticket",
                    "is_categorical": False,
                    "possible_values": []
                },
                "train id": {
                    "description": "id of the train",
                    "is_categorical": False
                },
                "choice": {
                    "description": "number of trains that meet the requirement",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "police": {
            "description": "find a police station for help",
            "slots": {
                "name": {
                    "description": "name of the police station",
                    "is_categorical": False,
                    "possible_values": []
                },
                "address": {
                    "description": "address of the police station",
                    "is_categorical": False,
                    "possible_values": []
                },
                "postcode": {
                    "description": "postcode of the police station",
                    "is_categorical": False,
                    "possible_values": []
                },
                "phone": {
                    "description": "phone number of the police station",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "hospital": {
            "description": "find a hospital for help",
            "slots": {
                "department": {
                    "description": "specific department of the hospital",
                    "is_categorical": False,
                    "possible_values": []
                },
                "address": {
                    "description": "address of the hospital",
                    "is_categorical": False,
                    "possible_values": []
                },
                "phone": {
                    "description": "phone number of the hospital",
                    "is_categorical": False,
                    "possible_values": []
                },
                "postcode": {
                    "description": "postcode of the hospital",
                    "is_categorical": False,
                    "possible_values": []
                }
            }
        },
        "general": {
            "description": "general domain without slots",
            "slots": {}
        }
    },
    "intents": {
        "inform": {
            "description": "inform the value of a slot"
        },
        "request": {
            "description": "ask for the value of a slot"
        },
        "nobook": {
            "description": "inform the user that the booking is failed"
        },
        "reqmore": {
            "description": "ask the user for more instructions"
        },
        "book": {
            "description": "book something for the user"
        },
        "bye": {
            "description": "say goodbye to the user and end the conversation"
        },
        "thank": {
            "description": "thanks for the help"
        },
        "welcome": {
            "description": "you're welcome"
        },
        "greet": {
            "description": "express greeting"
        },
        "recommend": {
            "description": "recommend a choice to the user"
        },
        "select": {
            "description": "provide several choices for the user"
        },
        "offerbook": {
            "description": "ask the user if he or she needs booking"
        },
        "offerbooked": {
            "description": "provide information about the booking"
        },
        "nooffer": {
            "description": "inform the user that there is no result satisfies user requirements"
        }
    },
    "state": {
        "attraction": {
            "type": "",
            "name": "",
            "area": ""
        },
        "hotel": {
            "name": "",
            "area": "",
            "parking": "",
            "price range": "",
            "stars": "",
            "internet": "",
            "type": "",
            "book stay": "",
            "book day": "",
            "book people": ""
        },
        "restaurant": {
            "food": "",
            "price range": "",
            "name": "",
            "area": "",
            "book time": "",
            "book day": "",
            "book people": ""
        },
        "taxi": {
            "leave at": "",
            "destination": "",
            "departure": "",
            "arrive by": ""
        },
        "train": {
            "leave at": "",
            "destination": "",
            "day": "",
            "arrive by": "",
            "departure": "",
            "book people": ""
        },
        "hospital": {
            "department": ""
        }
    },
    "dialogue_acts": {
        "categorical": {},
        "non-categorical": {},
        "binary": {}
    }
}

slot_name_map = {
    'addr': "address",
    'post': "postcode",
    'pricerange': "price range",
    'arrive': "arrive by",
    'arriveby': "arrive by",
    'leave': "leave at",
    'leaveat': "leave at",
    'depart': "departure",
    'dest': "destination",
    'fee': "entrance fee",
    'open': 'open hours',
    'car': "type",
    'car type': "type",
    'ticket': 'price',
    'trainid': 'train id',
    'id': 'train id',
    'people': 'book people',
    'stay': 'book stay',
    'none': '',
    'attraction': {
        'price': 'entrance fee'
    },
    'hospital': {},
    'hotel': {
        'day': 'book day', 'price': "price range"
    },
    'restaurant': {
        'day': 'book day', 'time': 'book time', 'price': "price range"
    },
    'taxi': {},
    'train': {
        'day': 'day', 'time': "duration"
    },
    'police': {},
    'booking': {}
}

reverse_da_slot_name_map = {
    'address': 'Addr',
    'postcode': 'Post',
    'price range': 'Price',
    'arrive by': 'Arrive',
    'leave at': 'Leave',
    'departure': 'Depart',
    'destination': 'Dest',
    'entrance fee': 'Fee',
    'open hours': 'Open',
    'price': 'Ticket',
    'train id': 'Id',
    'book people': 'People',
    'book stay': 'Stay',
    'book day': 'Day',
    'book time': 'Time',
    'duration': 'Time',
    'taxi': {
        'type': 'Car',
        'phone': 'Phone'
    }
}

digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
}

cnt_domain_slot = Counter()


class BookingActRemapper:

    def __init__(self, ontology):
        self.ontology = ontology
        self.reset()

    def reset(self):
        self.current_domains_user = []
        self.current_domains_system = []
        self.booked_domains = []

    def retrieve_current_domain_from_user(self, turn_id, ori_dialog):
        prev_user_turn = ori_dialog[turn_id - 1]

        dialog_acts = prev_user_turn.get('dialog_act', [])
        keyword_domains_user = get_keyword_domains(prev_user_turn)
        current_domains_temp = get_current_domains_from_act(dialog_acts)
        self.current_domains_user = current_domains_temp if current_domains_temp else self.current_domains_user
        next_user_domains = get_next_user_act_domains(ori_dialog, turn_id)

        return keyword_domains_user, next_user_domains

    def retrieve_current_domain_from_system(self, turn_id, ori_dialog):

        system_turn = ori_dialog[turn_id]
        dialog_acts = system_turn.get('dialog_act', [])
        keyword_domains_system = get_keyword_domains(system_turn)
        current_domains_temp = get_current_domains_from_act(dialog_acts)
        self.current_domains_system = current_domains_temp if current_domains_temp else self.current_domains_system
        booked_domain_current = self.check_domain_booked(system_turn)

        return keyword_domains_system, booked_domain_current

    def remap(self, turn_id, ori_dialog):

        keyword_domains_user, next_user_domains = self.retrieve_current_domain_from_user(
            turn_id, ori_dialog)
        keyword_domains_system, booked_domain_current = self.retrieve_current_domain_from_system(
            turn_id, ori_dialog)

        # only need to remap if there is a dialog action labelled
        dialog_acts = ori_dialog[turn_id].get('dialog_act', [])
        spans = ori_dialog[turn_id].get('span_info', [])
        if dialog_acts:

            flattened_acts = flatten_acts(dialog_acts)
            flattened_spans = flatten_span_acts(spans)
            remapped_acts, error_local = remap_acts(flattened_acts, self.current_domains_user,
                                                    booked_domain_current, keyword_domains_user,
                                                    keyword_domains_system, self.current_domains_system,
                                                    next_user_domains, self.ontology)

            remapped_spans, _ = remap_acts(flattened_spans, self.current_domains_user,
                                           booked_domain_current, keyword_domains_user,
                                           keyword_domains_system, self.current_domains_system,
                                           next_user_domains, self.ontology)

            deflattened_remapped_acts = deflat_acts(remapped_acts)
            deflattened_remapped_spans = deflat_span_acts(remapped_spans)

            return deflattened_remapped_acts, deflattened_remapped_spans
        else:
            return dialog_acts, spans

    def check_domain_booked(self, turn):

        booked_domain_current = None
        return booked_domain_current

        # workaround
        for domain in turn['metadata']:
            if turn['metadata'][domain]["book"]["booked"] and domain not in self.booked_domains:
                booked_domain_current = domain.capitalize()
                self.booked_domains.append(domain)
        return booked_domain_current


def get_keyword_domains(turn):
    keyword_domains = []
    text = turn['text']
    for d in ["Hotel", "Restaurant", "Train"]:
        if d.lower() in text.lower():
            keyword_domains.append(d)
    return keyword_domains


def get_current_domains_from_act(dialog_acts):

    current_domains_temp = []
    for dom_int in dialog_acts:
        domain, intent = dom_int.split('-')
        if domain in ["general", "Booking"]:
            continue
        if domain not in current_domains_temp:
            current_domains_temp.append(domain)

    return current_domains_temp


def get_next_user_act_domains(ori_dialog, turn_id):
    domains = []
    try:
        next_user_act = ori_dialog[turn_id + 1]['dialog_act']
        domains = get_current_domains_from_act(next_user_act)
    except:
        # will fail if system act is the last act of the dialogue
        pass
    return domains


def flatten_acts(dialog_acts):
    flattened_acts = []
    for dom_int in dialog_acts:
        domain, intent = dom_int.split('-')
        for slot_value in dialog_acts[dom_int]:
            slot = slot_value[0]
            value = slot_value[1]
            flattened_acts.append((domain, intent, slot, value))

    return flattened_acts


def flatten_span_acts(span_acts):

    flattened_acts = []
    for span_act in span_acts:
        domain, intent = span_act[0].split("-")
        flattened_acts.append((domain, intent, span_act[1], span_act[2:]))
    return flattened_acts


def deflat_acts(flattened_acts):

    dialog_acts = dict()

    for act in flattened_acts:
        domain, intent, slot, value = act
        if f"{domain}-{intent}" not in dialog_acts.keys():
            dialog_acts[f"{domain}-{intent}"] = [[slot, value]]
        else:
            dialog_acts[f"{domain}-{intent}"].append([slot, value])

    return dialog_acts


def deflat_span_acts(flattened_acts):

    dialog_span_acts = []
    for act in flattened_acts:
        domain, intent, slot, value = act
        if value == 'none':
            continue
        new_act = [f"{domain}-{intent}", slot]
        new_act.extend(value)
        dialog_span_acts.append(new_act)

    return dialog_span_acts


def remap_acts(flattened_acts, current_domains, booked_domain=None, keyword_domains_user=None,
               keyword_domains_system=None, current_domain_system=None, next_user_domain=None, ontology=None):

    # We now look for all cases that can happen: Booking domain, Booking within a domain or taxi-inform-car for booking
    error = 0
    remapped_acts = []

    # if there is more than one current domain or none at all, we try to get booked domain differently
    if len(current_domains) != 1 and booked_domain:
        current_domains = [booked_domain]
    elif len(current_domains) != 1 and len(keyword_domains_user) == 1:
        current_domains = keyword_domains_user
    elif len(current_domains) != 1 and len(keyword_domains_system) == 1:
        current_domains = keyword_domains_system
    elif len(current_domains) != 1 and len(current_domain_system) == 1:
        current_domains = current_domain_system
    elif len(current_domains) != 1 and len(next_user_domain) == 1:
        current_domains = next_user_domain

    for act in flattened_acts:
        try:
            domain, intent, slot, value = act
            if f"{domain}-{intent}-{slot}" == "Booking-Book-Ref":
                # We need to remap that booking act now
                potential_domain = current_domains[0]
                remapped_acts.append(
                    (potential_domain, "Book", "none", "none"))
                if ontology_check(potential_domain, slot, ontology):
                    remapped_acts.append(
                        (potential_domain, "Inform", "Ref", value))
            elif domain == "Booking" and intent == "Book" and slot != "Ref":
                # the book intent is here actually an inform intent according to the data
                potential_domain = current_domains[0]
                if ontology_check(potential_domain, slot, ontology):
                    remapped_acts.append(
                        (potential_domain, "Inform", slot, value))
            elif domain == "Booking" and intent == "Inform":
                # the inform intent is here actually a request intent according to the data
                potential_domain = current_domains[0]
                if ontology_check(potential_domain, slot, ontology):
                    remapped_acts.append(
                        (potential_domain, "OfferBook", slot, value))
            elif domain == "Booking" and intent in ["NoBook", "Request"]:
                potential_domain = current_domains[0]
                if ontology_check(potential_domain, slot, ontology):
                    remapped_acts.append(
                        (potential_domain, intent, slot, value))
            elif f"{domain}-{intent}-{slot}" == "Taxi-Inform-Car":
                # taxi-inform-car actually triggers the booking and informs on a car
                remapped_acts.append((domain, "Book", "none", "none"))
                remapped_acts.append((domain, intent, slot, value))
            elif f"{domain}-{intent}-{slot}" in ["Train-Inform-Ref", "Train-OfferBooked-Ref"]:
                # train-inform/offerbooked-ref actually triggers the booking and informs on the reference number
                remapped_acts.append((domain, "Book", "none", "none"))
                remapped_acts.append((domain, "Inform", slot, value))
            elif domain == "Train" and intent == "OfferBooked" and slot != "Ref":
                # this is actually an inform act
                remapped_acts.append((domain, "Inform", slot, value))
            else:
                remapped_acts.append(act)
        except Exception as e:
            print("Error detected:", e)
            error += 1

    return remapped_acts, error


def ontology_check(domain_, slot_, init_ontology):

    domain = domain_.lower()
    slot = slot_.lower()
    if slot not in init_ontology['domains'][domain]['slots']:
        if slot in slot_name_map:
            slot = slot_name_map[slot]
        elif slot in slot_name_map[domain]:
            slot = slot_name_map[domain][slot]
    return slot in init_ontology['domains'][domain]['slots']


def reverse_da(dialogue_acts):
    global reverse_da_slot_name_map
    das = {}
    for da_type in dialogue_acts:
        for da in dialogue_acts[da_type]:
            intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get(
                'value', '')
            if domain == 'general':
                Domain_Intent = '-'.join([domain, intent])
            elif intent == 'nooffer':
                Domain_Intent = '-'.join([domain.capitalize(), 'NoOffer'])
            elif intent == 'nobook':
                Domain_Intent = '-'.join([domain.capitalize(), 'NoBook'])
            elif intent == 'offerbook':
                Domain_Intent = '-'.join([domain.capitalize(), 'OfferBook'])
            else:
                Domain_Intent = '-'.join([domain.capitalize(),
                                         intent.capitalize()])
            das.setdefault(Domain_Intent, [])
            if slot in reverse_da_slot_name_map:
                Slot = reverse_da_slot_name_map[slot]
            elif domain in reverse_da_slot_name_map and slot in reverse_da_slot_name_map[domain]:
                Slot = reverse_da_slot_name_map[domain][slot]
            else:
                Slot = slot.capitalize()
            if value == '':
                if intent == 'request':
                    value = '?'
                else:
                    value = 'none'
            if Slot == '':
                Slot = 'none'
            das[Domain_Intent].append([Slot, value])
    return das


def normalize_domain_slot_value(domain, slot, value):
    global ontology, slot_name_map
    domain = domain.lower()
    slot = slot.lower()
    value = value.strip()
    if value in ['do nt care', "do n't care"]:
        value = 'dontcare'
    if value in ['?', 'none', 'not mentioned']:
        value = ""
    if domain not in ontology['domains']:
        raise Exception(f'{domain} not in ontology')
    if slot not in ontology['domains'][domain]['slots']:
        if slot in slot_name_map:
            slot = slot_name_map[slot]
        elif slot in slot_name_map[domain]:
            slot = slot_name_map[domain][slot]
        else:
            raise Exception(f'{domain}-{slot} not in ontology')
    assert slot == '' or slot in ontology['domains'][domain][
        'slots'], f'{(domain, slot, value)} not in ontology'
    return domain, slot, value


def convert_da(da_dict, utt, sent_tokenizer, word_tokenizer):
    '''
    convert multiwoz dialogue acts to required format
    :param da_dict: dict[(intent, domain, slot, value)] = [word_start, word_end]
    :param utt: user or system utt
    '''
    global ontology, digit2word, cnt_domain_slot

    converted_da = {
        'categorical': [],
        'non-categorical': [],
        'binary': []
    }
    sentences = sent_tokenizer.tokenize(utt)
    sent_spans = sent_tokenizer.span_tokenize(utt)
    tokens = [
        token for sent in sentences for token in word_tokenizer.tokenize(sent)]
    token_spans = [(sent_span[0] + token_span[0], sent_span[0] + token_span[1]) for sent, sent_span in
                   zip(sentences, sent_spans) for token_span in word_tokenizer.span_tokenize(sent)]
    # assert len(tokens) == len(token_spans)
    # for token, span in zip(tokens, token_spans):
    #     if utt[span[0]:span[1]] != '"':
    #         assert utt[span[0]:span[1]] == token

    for (intent, domain, slot, value), span in da_dict.items():
        if intent == 'request' or slot == '' or value == '':
            # binary dialog acts
            assert value == ''
            converted_da['binary'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot
            })
        elif ontology['domains'][domain]['slots'][slot]['is_categorical']:
            # categorical dialog acts
            converted_da['categorical'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot,
                'value': value
            })
        else:
            # non-categorical dialog acts
            converted_da['non-categorical'].append({
                'intent': intent,
                'domain': domain,
                'slot': slot,
                'value': value
            })
            # correct some value and try to give char level span
            match = False
            value = value.lower()
            if span and span[0] <= span[1]:
                # use original span annotation, but tokenizations are different
                start_word, end_word = span
                if end_word >= len(tokens):
                    # due to different tokenization, sometimes will out of index
                    delta = end_word - len(tokens) + 1
                    start_word -= delta
                    end_word -= delta
                start_char, end_char = token_spans[start_word][0], token_spans[end_word][1]
                value_span = utt[start_char:end_char].lower()
                match = True
                if value_span == value:
                    cnt_domain_slot['span match'] += 1
                elif value.isdigit() and value in digit2word and digit2word[value] == value_span:
                    # !!!CHANGE VALUE: value is digit but value span is word
                    cnt_domain_slot['digit value match'] += 1
                elif ''.join(value.split()) == ''.join(value_span.split()):
                    # !!!CHANGE VALUE: equal when remove blank
                    cnt_domain_slot['remove blank'] += 1
                elif value in value_span:
                    # value in value_span
                    start_char += value_span.index(value)
                    end_char = start_char + len(value)
                    assert utt[start_char:end_char].lower(
                    ) == value, f'{[value, utt[start_char:end_char], utt]}'
                    cnt_domain_slot['value in span'] += 1
                elif ':' in value and value == '0' + value_span:
                    # !!!CHANGE VALUE: time x:xx == 0x:xx
                    cnt_domain_slot['x:xx == 0x:xx'] += 1
                else:
                    # span mismatch, search near 1-2 words
                    for window in range(1, 3):
                        start = max(0, start_word - window)
                        end = min(len(token_spans) - 1, end_word + window)
                        large_span = utt[token_spans[start]
                                         [0]:token_spans[end][1]].lower()
                        if value in large_span:
                            start_char = token_spans[start][0] + \
                                large_span.index(value)
                            end_char = start_char + len(value)
                            assert utt[
                                start_char:end_char].lower() == value, f'{[value, utt[start_char:end_char], utt]}'
                            cnt_domain_slot[f'window={window}'] += 1
                            break
                    else:
                        # still not found
                        match = False

            if match:
                converted_da['non-categorical'][-1]['value'] = utt[start_char:end_char]
                converted_da['non-categorical'][-1]['start'] = start_char
                converted_da['non-categorical'][-1]['end'] = end_char
                cnt_domain_slot['have span'] += 1
            else:
                cnt_domain_slot['no span'] += 1
    return converted_da


def preprocess():
    original_data_dir = 'emowoz'
    new_data_dir = 'data'

    if not os.path.exists(original_data_dir):
        original_data_zip = 'MultiWOZ_2.1.zip'
        if not os.path.exists(original_data_zip):
            raise FileNotFoundError(
                f'cannot find original data {original_data_zip} in multiwoz21/, should manually download MultiWOZ_2.1.zip from https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip')
        else:
            archive = ZipFile(original_data_zip)
            archive.extractall()

    os.makedirs(new_data_dir, exist_ok=True)
    for filename in os.listdir(original_data_dir):
        if 'db' in filename:
            copy2(f'{original_data_dir}/{filename}', new_data_dir)

    # how about emowoz-dialmage
    original_data = json.load(
        open(f'{original_data_dir}/emowoz-multiwoz.json'))
    global ontology, cnt_domain_slot

    data_split = json.load(open(f'{original_data_dir}/data_split.json'))
    val_list = data_split["dev"]["multiwoz"]
    test_list = data_split["test"]["multiwoz"]
    # val_list = set(open(f'{original_data_dir}/valListFile.txt').read().split())
    # test_list = set(open(f'{original_data_dir}/testListFile.txt').read().split())
    dataset = 'multiwoz21'
    splits = ['train', 'validation', 'test']
    dialogues_by_split = {split: [] for split in splits}
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()
    booking_remapper = BookingActRemapper(ontology)
    for ori_dialog_id, ori_dialog in tqdm(original_data.items()):
        if ori_dialog_id in val_list:
            split = 'validation'
        elif ori_dialog_id in test_list:
            split = 'test'
        else:
            split = 'train'
        dialogue_id = f'{dataset}-{split}-{len(dialogues_by_split[split])}'

        # get user goal and involved domains
        cur_domains = []

        dialogue = {
            'dataset': dataset,
            'data_split': split,
            'dialogue_id': dialogue_id,
            'original_id': ori_dialog_id,
            'domains': cur_domains,  # will be updated by dialog_acts and state
            'goal': "",
            'turns': []
        }

        booking_remapper.reset()
        belief_domains = ['attraction', 'restaurant',
                          'train', 'hotel', 'taxi', 'hospital']
        entity_booked_dict = dict((domain, False) for domain in belief_domains)
        for turn_id, turn in enumerate(ori_dialog['log']):
            # correct some grammar errors in the text, mainly following `tokenization.md` in MultiWOZ_2.1
            text = turn['text']
            text = re.sub(" Im ", " I'm ", text)
            text = re.sub(" im ", " i'm ", text)
            text = re.sub(r"^Im ", "I'm ", text)
            text = re.sub(r"^im ", "i'm ", text)
            text = re.sub("theres", "there's", text)
            text = re.sub("dont", "don't", text)
            text = re.sub("whats", "what's", text)
            text = re.sub('thats', "that's", text)
            utt = text
            speaker = 'user' if turn_id % 2 == 0 else 'system'

            das = turn.get('dialog_act', [])
            spans = turn.get('span_info', [])

            if speaker == 'system':
                das, spans = booking_remapper.remap(turn_id, ori_dialog['log'])

            da_dict = {}
            # transform DA
            for Domain_Intent in das:
                domain, intent = Domain_Intent.lower().split('-')
                assert intent in ontology['intents'], f'{ori_dialog_id}:{turn_id}:da\t{intent} not in ontology'
                for Slot, value in das[Domain_Intent]:
                    domain, slot, value = normalize_domain_slot_value(
                        domain, Slot, value)
                    if domain not in cur_domains:
                        # update original cur_domains
                        cur_domains.append(domain)
                    da_dict[(intent, domain, slot, value,)] = []

            for span in spans:
                Domain_Intent, Slot, value, start_word, end_word = span
                domain, intent = Domain_Intent.lower().split('-')
                domain, slot, value = normalize_domain_slot_value(
                    domain, Slot, value)
                assert (intent, domain, slot, value,) in da_dict
                da_dict[(intent, domain, slot, value,)] = [
                    start_word, end_word]

            dialogue_acts = convert_da(
                da_dict, utt, sent_tokenizer, word_tokenizer)

            # reverse_das = reverse_da(dialogue_acts)
            # das_list = sorted([(Domain_Intent, Slot, ''.join(value.split()).lower()) for Domain_Intent in das for Slot, value in das[Domain_Intent]])
            # reverse_das_list = sorted([(Domain_Intent, Slot, ''.join(value.split()).lower()) for Domain_Intent in reverse_das for Slot, value in reverse_das[Domain_Intent]])
            # if das_list != reverse_das_list:
            #     print(das_list)
            #     print(reverse_das_list)
            #     print()
            #     print()

            dialogue['turns'].append({
                'speaker': speaker,
                'utterance': utt,
                'utt_idx': turn_id,
                'dialogue_acts': dialogue_acts,
                'emotion': turn['emotion']
            })

            # add to dialogue_acts dictionary in the ontology
            for da_type in dialogue_acts:
                das = dialogue_acts[da_type]
                for da in das:
                    ontology["dialogue_acts"][da_type].setdefault(
                        (da['intent'], da['domain'], da['slot']), {})
                    ontology["dialogue_acts"][da_type][(
                        da['intent'], da['domain'], da['slot'])][speaker] = True

            if speaker == 'system':
                # add state to last user turn
                # add empty db_results
                # turn_state = turn['metadata']
                cur_state = copy.deepcopy(ontology['state'])
                booked = {}
                # for domain in turn_state:
                #     if domain not in cur_state:
                #         continue
                #     for subdomain in ['semi', 'book']:
                #         for slot, value in turn_state[domain][subdomain].items():
                #             if slot == 'ticket':
                #                 continue
                #             elif slot == 'booked':
                #                 assert domain in ontology['domains']
                #                 booked[domain] = value
                #                 continue
                #             _, slot, value = normalize_domain_slot_value(
                #                 domain, slot, value)
                #             cur_state[domain][slot] = value
                dialogue['turns'][-2]['state'] = cur_state
                # entity_booked_dict, booked = fix_entity_booked_info(
                #     entity_booked_dict, booked)
                dialogue['turns'][-1]['booked'] = booked
        dialogues_by_split[split].append(dialogue)
    # pprint(cnt_domain_slot.most_common())
    dialogues = []
    for split in splits:
        dialogues += dialogues_by_split[split]
    for da_type in ontology['dialogue_acts']:
        ontology["dialogue_acts"][da_type] = sorted([str(
            {'user': speakers.get('user', False), 'system': speakers.get('system', False), 'intent': da[0],
             'domain': da[1], 'slot': da[2]}) for da, speakers in ontology["dialogue_acts"][da_type].items()])
    json.dump(dialogues[:10], open(f'dummy_data.json', 'w',
              encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(ontology, open(f'{new_data_dir}/ontology.json',
              'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dialogues, open(f'{new_data_dir}/dialogues.json',
              'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with ZipFile('data.zip', 'w', ZIP_DEFLATED) as zf:
        for filename in os.listdir(new_data_dir):
            zf.write(f'{new_data_dir}/{filename}')
    # rmtree(original_data_dir)
    # rmtree(new_data_dir)
    return dialogues, ontology


def fix_entity_booked_info(entity_booked_dict, booked):
    for domain in entity_booked_dict:
        if not entity_booked_dict[domain] and booked[domain]:
            entity_booked_dict[domain] = True
            booked[domain] = []
    return entity_booked_dict, booked


if __name__ == '__main__':
    preprocess()
