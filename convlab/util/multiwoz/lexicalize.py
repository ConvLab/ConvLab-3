from copy import deepcopy

from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from convlab.util import relative_import_module_from_unified_datasets

reverse_da_slot_name_map = relative_import_module_from_unified_datasets('multiwoz21', 'preprocess.py', 'reverse_da_slot_name_map')

def delexicalize_da(da, requestable):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if slot == "":
            slot = 'none'
        if intent in requestable:
            v = '?'
        else:
            if slot == 'none':
                v = 'none'
            else:
                k = '-'.join([intent, domain, slot])
                counter.setdefault(k, 0)
                counter[k] += 1
                v = str(counter[k])
        delexicalized_da.append([domain, intent, slot, v])
    return delexicalized_da


def flat_da(delexicalized_da):
    flaten = ['-'.join(x) for x in delexicalized_da]
    return flaten


def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = {}
    for da in meta:
        d, i, s, v = da.split('-')
        k = '-'.join((d, i))
        if k not in dialog_act:
            dialog_act[k] = []
        dialog_act[k].append([s, v])
    return dialog_act


def lexicalize_da(meta, entities, state, requestable):
    meta = deepcopy(meta)
    for k, v in meta.items():
        domain, intent = k.split('-')
        if domain in ['general']:
            continue
        elif intent in requestable:
            for pair in v:
                pair[1] = '?'
        else:
            if intent == "book":
                # this means we booked something. We retrieve reference number here
                for pair in v:
                    n = int(pair[1]) - 1 if pair[1] != 'none' else 0
                    if len(entities[domain]) > n:
                        if 'Ref' in entities[domain][n]:
                            pair[1] = entities[domain][n]['Ref']
                continue

            for pair in v:
                if pair[1] == 'none':
                    continue
                elif pair[0].lower() == 'choice':
                    pair[1] = str(len(entities[domain]))
                else:
                    # try to retrieve value from the database entity, otherwise from the belief state
                    if domain != 'taxi':
                        slot_reverse = reverse_da_slot_name_map.get(pair[0], pair[0])
                    else:
                        slot_reverse = reverse_da_slot_name_map['taxi'].get(pair[0], pair[0])
                    slot_old = REF_SYS_DA[domain.capitalize()].get(slot_reverse, pair[0].lower())
                    slot = pair[0]
                    n = int(pair[1]) - 1
                    if len(entities[domain]) > n:
                        if slot in entities[domain][n]:
                            pair[1] = entities[domain][n][slot]
                        elif "".join(slot.split(" ")) in entities[domain][n]:
                            pair[1] = entities[domain][n]["".join(slot.split(" "))]
                        elif slot.capitalize() in entities[domain][n]:
                            pair[1] = entities[domain][n][slot.capitalize()]
                        elif slot_old in entities[domain][n]:
                            pair[1] = entities[domain][n][slot_old]
                        elif slot in state[domain]:
                            pair[1] = state[domain][slot]
                        else:
                            pair[1] = 'not available'
                    elif slot in state[domain]:
                        pair[1] = state[domain][slot] if state[domain][slot] else 'none'
                    else:
                        pair[1] = 'none'

    tuples = []
    for domain_intent, svs in meta.items():
        for slot, value in svs:
            domain, intent = domain_intent.split('-')
            tuples.append([intent, domain, slot, value])
    return tuples
