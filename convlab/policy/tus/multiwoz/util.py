from convlab.policy.tus.multiwoz.Da2Goal import SysDa2Goal, UsrDa2Goal
import json

NOT_MENTIONED = "not mentioned"


def int2onehot(index, output_dim=6, remove_zero=False):
    one_hot = [0] * output_dim
    if remove_zero:
        if index == 0:
            one_hot[index] = 1
    else:
        if index >= 0:
            one_hot[index] = 1

    return one_hot


def parse_user_goal(user_goal):
    """flatten user goal structure"""
    goal = user_goal.domain_goals
    user_goal = {}
    for domain in goal:
        if domain not in UsrDa2Goal:
            continue
        for slot_type in goal[domain]:
            if slot_type in ["fail_info", "fail_book", "booked"]:
                continue  # TODO [fail_info] fix in the future
            if slot_type in ["info", "book", "reqt"]:
                for slot in goal[domain][slot_type]:
                    slot_name = f"{domain}-{slot.lower()}"
                    user_goal[slot_name] = goal[domain][slot_type][slot]

    return user_goal


def parse_dialogue_act(dialogue_act):
    """ transfer action from dict to list """
    actions = []
    for act in dialogue_act:
        domain, intent = act.split('-')
        for slot, value in dialogue_act[act]:
            value_dict = {"do nt care": "dontcare"}
            if value in value_dict:
                value = value_dict[value]
            actions.append([intent, domain, slot, value])

    return actions


def metadata2state(metadata):
    """
    parse metadata in the data set or dst
    """
    slot_value = {}

    for domain in metadata:
        for slot in metadata[domain]["semi"]:
            slot_name = f"{domain.lower()}-{slot.lower()}"
            value = metadata[domain]["semi"][slot]
            if not value or value == NOT_MENTIONED:
                value = "none"
            slot_value[slot_name] = value

        for slot in metadata[domain]["book"]:
            if slot == "booked":
                continue
            slot_name = f"{domain.lower()}-{slot.lower()}"
            value = metadata[domain]["book"][slot]
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


def act2slot(intent, domain, slot, value, all_values):

    if domain not in UsrDa2Goal:
        # print(f"Not handle domain {domain}")
        return ""

    if domain == "booking":
        slot = SysDa2Goal[domain][slot]
        domain = get_booking_domain(slot, value, all_values)
        return f"{domain}-{slot}"

    elif domain in UsrDa2Goal:
        if slot in SysDa2Goal[domain]:
            slot = SysDa2Goal[domain][slot]
        elif slot in UsrDa2Goal[domain]:
            slot = UsrDa2Goal[domain][slot]
        elif slot in SysDa2Goal["booking"]:
            slot = SysDa2Goal["booking"][slot]
        # else:
        #     print(
        #         f"UNSEEN ACTION IN GENERATE LABEL {intent, domain, slot, value}")

        return f"{domain}-{slot}"

    print("strange!!!")
    print(intent, domain, slot, value)

    return ""


def get_user_history(dialog, all_values):
    turn_num = len(dialog)
    mentioned_slot = []
    for turn_id in range(0, turn_num, 2):
        usr_act = parse_dialogue_act(
            dialog[turn_id]["dialog_act"])
        for intent, domain, slot, value in usr_act:
            slot_name = act2slot(
                intent, domain.lower(), slot.lower(), value.lower(), all_values)
            if slot_name not in mentioned_slot:
                mentioned_slot.append(slot_name)
    return mentioned_slot


def update_config_file(file_name, attribute, value):
    with open(file_name, 'r') as config_file:
        config = json.load(config_file)

    config[attribute] = value
    print(config)
    with open(file_name, 'w') as config_file:
        json.dump(config, config_file)
    print(f"update {attribute} = {value}")
