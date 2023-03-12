import os
import json

from convlab.policy.vector.vector_binary import VectorBinary
from convlab.util import load_ontology, load_database
from convlab.util.custom_util import timeout


def create_description_dicts(name='multiwoz21'):

    vector = VectorBinary(name)
    ontology = load_ontology(name)
    default_state = ontology['state']
    domains = list(ontology['domains'].keys())

    if name == "multiwoz21" or name == "crosswoz":
        db = load_database(name)
        db_domains = db.domains
    else:
        db = None
        db_domains = []

    da_voc = vector.da_voc
    da_voc_opp = vector.da_voc_opp

    description_dict_semantic = {}

    for domain in default_state:
        for slot in default_state[domain]:
            domain = domain.lower()
            description_dict_semantic[f"user goal-{domain}-{slot.lower()}"] = f"user goal {domain} {slot}"

    if db_domains:
        for domain in db_domains:
            domain = domain.lower()
            description_dict_semantic[f"db-{domain}-entities"] = f"data base {domain} number of entities"
            description_dict_semantic[f"general-{domain}-booked"] = f"general {domain} booked"

    for domain in domains:
        domain = domain.lower()
        description_dict_semantic[f"general-{domain}"] = f"domain {domain}"

    for act in da_voc:
        domain, intent, slot, value = act
        domain = domain.lower()
        act = "_".join(act)
        description_dict_semantic["system-"+act.lower()] = f"last system act {domain} {intent} {slot} {value}"

    for act in da_voc_opp:
        domain, intent, slot, value = [item.lower() for item in act]
        domain = domain.lower()
        act = "_".join(act)
        description_dict_semantic["user-"+act.lower()] = f"user act {domain} {intent} {slot} {value}"

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root_dir, "descriptions"), exist_ok=True)
    with open(os.path.join(root_dir, 'descriptions', f'semantic_information_descriptions_{name}.json'), "w") as f:
        json.dump(description_dict_semantic, f)


if __name__ == '__main__':
    create_description_dicts()
