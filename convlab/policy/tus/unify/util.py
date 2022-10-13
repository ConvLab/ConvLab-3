from convlab.policy.tus.multiwoz.Da2Goal import SysDa2Goal, UsrDa2Goal
from convlab.util import load_dataset

import json

NOT_MENTIONED = "not mentioned"


def load_experiment_dataset(data_name="multiwoz21", dial_ids_order=0, split2ratio=1):
    ratio = {'train': split2ratio, 'validation': split2ratio}
    if data_name == "all" or data_name == "sgd+tm" or data_name == "tm":
        print("merge all datasets...")
        if data_name == "all":
            all_dataset = ["multiwoz21", "sgd", "tm1", "tm2", "tm3"]
        if data_name == "sgd+tm":
            all_dataset = ["sgd", "tm1", "tm2", "tm3"]
        if data_name == "tm":
            all_dataset = ["tm1", "tm2", "tm3"]

        datasets = {}
        for name in all_dataset:
            datasets[name] = load_dataset(
                name,
                dial_ids_order=dial_ids_order,
                split2ratio=ratio)
        raw_data = merge_dataset(datasets, all_dataset[0])

    else:
        print(f"load single dataset {data_name}/{split2ratio}")
        raw_data = load_dataset(data_name,
                                dial_ids_order=dial_ids_order,
                                split2ratio=ratio)
    return raw_data


def merge_dataset(datasets, data_name):
    data_split = [x for x in datasets[data_name]]
    raw_data = {}
    for data_type in data_split:
        raw_data[data_type] = []
        for dataname, dataset in datasets.items():
            print(f"merge {dataname}...")
            raw_data[data_type] += dataset[data_type]
    return raw_data


def int2onehot(index, output_dim=6, remove_zero=False):
    one_hot = [0] * output_dim
    if remove_zero:
        if index == 0:
            one_hot[index] = 1
    else:
        if index >= 0:
            one_hot[index] = 1

    return one_hot


def parse_user_goal(raw_goal):
    """flatten user goal structure"""
    goal = raw_goal.domain_goals
    user_goal = {}
    for domain in goal:
        # if domain not in UsrDa2Goal:
        #     continue
        for slot_type in goal[domain]:
            if slot_type in ["fail_info", "fail_book", "booked"]:
                continue  # TODO [fail_info] fix in the future
            if slot_type in ["info", "book", "reqt"]:
                for slot in goal[domain][slot_type]:
                    slot_name = f"{domain}-{slot}"
                    user_goal[slot_name] = goal[domain][slot_type][slot]

    return user_goal


def parse_dialogue_act(dialogue_act):
    """ transfer action from dict to list """
    actions = []
    for action_type in dialogue_act:
        for act in dialogue_act[action_type]:
            domain = act["domain"]
            if "value" in act:
                actions.append(
                    [act["intent"], domain, act["slot"], act["value"]])
            else:
                if act["intent"] == "request":
                    actions.append(
                        [act["intent"], domain, act["slot"], "?"])
                else:
                    slot = act.get("slot", "none")
                    value = act.get("value", "none")
                    actions.append(
                        [act["intent"], domain, slot, value])

    return actions


def metadata2state(metadata):
    """
    parse metadata in the data set or dst
    """
    slot_value = {}

    for domain in metadata:
        for slot in metadata[domain]:
            slot_name = f"{domain}-{slot}"
            value = metadata[domain][slot]
            if not value or value == NOT_MENTIONED:
                value = "none"
            slot_value[slot_name] = value

    return slot_value


def get_booking_domain(slot, value, all_values, domain_list):
    """ 
    find the domain for domain booking, excluding slot "ref"
    """
    found = ""
    if not slot:
        return found
    slot = slot.lower()
    value = value.lower()
    for domain in domain_list:
        if slot in all_values["all_value"][domain] \
                and value in all_values["all_value"][domain][slot]:
            found = domain
    return found


def update_config_file(file_name, attribute, value):
    with open(file_name, 'r') as config_file:
        config = json.load(config_file)

    config[attribute] = value
    print(config)
    with open(file_name, 'w') as config_file:
        json.dump(config, config_file)
    print(f"update {attribute} = {value}")


def create_goal(dialog) -> list:
    # a list of {'intent': ..., 'domain': ..., 'slot': ..., 'value': ...}
    dicts = []
    for turn in dialog['turns']:
        # print(turn['speaker'])
        # assert (i % 2 == 0) == (turn['speaker'] == 'user')
        # if i % 2 == 0:
        if turn['speaker'] == 'user':
            dicts += turn['dialogue_acts']['categorical']
            dicts += turn['dialogue_acts']['binary']
            dicts += turn['dialogue_acts']['non-categorical']
    tuples = []
    for d in dicts:
        if "value" not in d:
            if d['intent'] == "request":
                value = "?"
            else:
                value = "none"
        else:
            value = d["value"]
        slot = d.get("slot", "none")
        domain = d['domain']  # .split('_')[0]
        tuples.append(
            (d['domain'], d['intent'], slot, value)
        )

    user_goal = []  # a list of (domain, intent, slot, value)
    for domain, intent, slot, value in tuples:
        # if slot == "intent":
        #     continue
        if not slot:
            continue
        if intent == "inform" and value == "":
            continue
        # if intent == "request" and value != "":
        #     intent = "inform"
        user_goal.append((domain, intent, slot, value))
    user_goal = unique_list(user_goal)
    inform_slots = {(domain, slot) for (domain, intent, slot,
                                        value) in user_goal if intent == "inform"}
    user_goal = [(domain, intent, slot, value) for (domain, intent, slot, value)
                 in user_goal if not (intent == "request" and (domain, slot) in inform_slots)]
    return user_goal


def unique_list(list_):
    r = []
    for x in list_:
        if x not in r:
            r.append(x)
    return r


def split_slot_name(slot_name):
    tokens = slot_name.split('-')
    if len(tokens) == 2:
        return tokens[0], tokens[1]
    else:
        return tokens[0], '-'.join(tokens[1:])


# copy from data.unified_datasets.multiwoz21
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

if __name__ == "__main__":
    print(split_slot_name("restaurant-search-location"))
    print(split_slot_name("sports-day.match"))
