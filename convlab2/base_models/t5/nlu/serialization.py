def serialize_dialogue_acts(dialogue_acts):
    da_dict = {}
    for da_type in dialogue_acts:
        for da in dialogue_acts[da_type]:
            intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
            intent_domain = f'[{intent}][{domain}]'
            da_dict.setdefault(intent_domain, [])
            da_dict[intent_domain].append(f'[{slot}][{value}]')
    return ';'.join([intent_domain+'('+','.join(slot_values)+')' for intent_domain, slot_values in da_dict.items()])

def deserialize_dialogue_acts(das_seq):
    dialogue_acts = []
    if len(das_seq) == 0:
        return dialogue_acts
    da_seqs = das_seq.split(']);[')  # will consume "])" and "["
    for i, da_seq in enumerate(da_seqs):
        if len(da_seq) == 0 or len(da_seq.split(']([')) != 2:
            continue
        if i == 0:
            if da_seq[0] == '[':
                da_seq = da_seq[1:]
        if i == len(da_seqs) - 1:
            if da_seq[-2:] == '])':
                da_seq = da_seq[:-2]
        
        try:
            intent_domain, slot_values = da_seq.split(']([')
            intent, domain = intent_domain.split('][')
        except:
            continue
        for slot_value in slot_values.split('],['):
            try:
                slot, value = slot_value.split('][')
            except:
                continue
            dialogue_acts.append({'intent': intent, 'domain': domain, 'slot': slot, 'value': value})
        
    return dialogue_acts

def equal_da_seq(dialogue_acts, das_seq):
    predict_dialogue_acts = deserialize_dialogue_acts(das_seq)
    das = sorted([(da['intent'], da['domain'], da['slot'], da.get('value', '')) for da_type in ['binary', 'categorical', 'non-categorical'] for da in dialogue_acts[da_type]])
    predict_das = sorted([(da['intent'], da['domain'], da['slot'], da.get('value', '')) for da in predict_dialogue_acts])
    if das != predict_das:
        return False
    return True
