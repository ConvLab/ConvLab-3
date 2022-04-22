import os
import json

from convlab2.policy.vector.vector_binary import VectorBinary
from convlab2.util import load_ontology, load_database


def create_description_dicts(name='multiwoz21'):

    vector = VectorBinary(name)
    ontology = load_ontology(name)
    default_state = ontology['state']
    domains = list(ontology['domains'].keys())
    db = load_database(name)
    db_domains = db.domains

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voc_file = os.path.join(root_dir, f'vector/action_dicts/{name}_VectorBinary/sys_da_voc.txt')
    voc_opp_file = os.path.join(root_dir, f'vector/action_dicts/{name}_VectorBinary/user_da_voc.txt')

    with open(voc_file) as f:
        da_voc = f.read().splitlines()
    with open(voc_opp_file) as f:
        da_voc_opp = f.read().splitlines()

    description_dict_semantic = {}

    for domain in default_state:
        for slot in default_state[domain]:
            description_dict_semantic[f"user goal-{domain}-{slot.lower()}"] = f"user goal {domain} {slot}"

    for domain in db_domains:
        description_dict_semantic[f"db-{domain}-entities"] = f"data base {domain} number of entities"
        description_dict_semantic[f"general-{domain}-booked"] = f"general {domain} booked"

    for domain in domains:
        description_dict_semantic[f"general-{domain}"] = f"domain {domain}"

    for act in da_voc:
        domain, intent, slot, value = act.split("-")
        description_dict_semantic["system-"+act.lower()] = f"last system act {domain} {intent} {slot} {value}"

    for act in da_voc_opp:
        domain, intent, slot, value = [item.lower() for item in act.split("-")]
        description_dict_semantic["user-"+act.lower()] = f"user act {domain} {intent} {slot} {value}"

    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, 'semantic_information_descriptions.json'), "w") as f:
        json.dump(description_dict_semantic, f)


if __name__ == '__main__':
    create_description_dicts()
