import os
import re
import json
from tqdm import tqdm

from convlab.util.unified_datasets_util import load_ontology, load_database


ontology = load_ontology('multiwoz21')
database = load_database('multiwoz21')

db = database.dbs

val2ds_dict = {}
ignore_slots = ['id', 'location', 'introduction', 'signature']
for domain in db:
    for entity in db[domain]:
        if isinstance(entity, dict):
            for slot in entity:
                if slot in ignore_slots:
                    continue
                if entity[slot] == '?':
                    continue
                # slot_name = f'[{domain}_{slot}]'.strip().lower().replace(' ', '')
                slot_name = f'[{slot}]'.strip().lower().replace(' ', '')
                value = str(entity[slot]).strip().lower()
                if value in ['yes', 'no']:
                    continue
                val2ds_dict[value] = slot_name
        elif isinstance(entity, str):
            if isinstance(db[domain][entity], list):
                for v in db[domain][entity]:
                    if entity == 'taxi_types':
                        entity = 'type'
                    val2ds_dict[v] = f'[{entity}]'
            else:
                print(type(entity))
                print(db[domain][entity])

for domain_name in ontology['domains']:
    domain = ontology['domains'][domain_name]
    for slot_name in domain['slots']:
        slot = domain['slots'][slot_name]
        if 'possible_values' not in slot:
            continue
        possible_vals = slot['possible_values']
        if len(possible_vals) > 0:
            for val in possible_vals:
                val2ds_dict[val] = f'[{slot_name}]'

def delex_and_norm(utt):
    norm_utt = utt.strip().lower()
    for val in val2ds_dict:
        norm_val = val.strip().lower()

        if val.isdigit() and len(val) == 1:
            if f' {norm_val} ' in f'{norm_utt}':
                norm_utt = norm_utt.replace(norm_val, f' {val2ds_dict[val]} ')
        else:
            if f'{norm_val}' in f'{norm_utt}':
                norm_utt = norm_utt.replace(norm_val, val2ds_dict[val])
    refs = re.match("^[0-9]{8}$", norm_utt)
    if refs:
        for r in refs:
            norm_utt = norm_utt.replace(r, '[ref]')
    refs = re.match("^[0-9]{10}$", norm_utt)
    if refs:
        for r in refs:
            norm_utt = norm_utt.replace(r, '[phone]')
    return norm_utt


for sys_name in ['emoloop', 'emoloop_base', 'emoloop_express', 'emoloop_recognise']:
    for seed in ['0', '1', '2', '3', '4', '5']:
        log_f = f'_{sys_name}/seed-{seed}.json'
        # log_f = 'test-result.json'
        if not os.path.exists(log_f):
            print(f'{log_f} does not exist. Skip')
            continue
        sys_log = json.load(open(log_f, 'r'))
        print(f'processing {log_f}')
        predictions = {}
        i = 0
        for item in tqdm(sys_log):
            dialog_id, _ = item['utt_idx'].split('_')
            dialog_id = dialog_id.replace('.json', '').lower()
            if dialog_id not in predictions:
                predictions[dialog_id] = []
            utt_delex = delex_and_norm(item['sys_utt'])
            # utt_delex = item['sys_utt']

            state = {}
            for domain in item['state']['belief_state']:
                state[domain] = {}
                for slot in item['state']['belief_state'][domain]:
                    state[domain][slot.replace(' ', '')] = item['state']['belief_state'][domain][slot]

            item_dict = {'response': utt_delex, 'state': state}
            predictions[dialog_id].append(item_dict)

        out_f = f'_{sys_name}/predictions-{seed}.json'
        # out_f = 'out.json'
        json.dump(predictions, open(out_f, 'w'), indent=4)
        # exit()