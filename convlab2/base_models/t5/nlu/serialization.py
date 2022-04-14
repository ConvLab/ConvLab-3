def serialize_dialogue_acts(dialogue_acts):
    da_seqs = []
    for da_type in dialogue_acts:
        for da in dialogue_acts[da_type]:
            intent, domain, slot = da['intent'], da['domain'], da['slot']
            if da_type == 'binary':
                da_seq = f'[{da_type}][{intent}][{domain}][{slot}]'
            else:
                value = da['value']
                da_seq = f'[{da_type}][{intent}][{domain}][{slot}][{value}]'
            da_seqs.append(da_seq)
    return ';'.join(da_seqs)

def deserialize_dialogue_acts(das_seq):
    dialogue_acts = {'binary': [], 'categorical': [], 'non-categorical': []}
    if len(das_seq) == 0:
        return dialogue_acts
    da_seqs = das_seq.split('];[')
    for i, da_seq in enumerate(da_seqs):
        if len(da_seq) == 0:
            continue
        if i == 0:
            if da_seq[0] == '[':
                da_seq = da_seq[1:]
        if i == len(da_seqs) - 1:
            if da_seq[-1] == ']':
                da_seq = da_seq[:-1]
        da = da_seq.split('][')
        if len(da) == 0:
            continue
        da_type = da[0]
        if len(da) == 5 and da_type in ['categorical', 'non-categorical']:
            dialogue_acts[da_type].append({'intent': da[1], 'domain': da[2], 'slot': da[3], 'value': da[4]})
        elif len(da) == 4 and da_type == 'binary':
            dialogue_acts[da_type].append({'intent': da[1], 'domain': da[2], 'slot': da[3]})
        else:
            # invalid da format, skip
            # print(das_seq)
            # print(da_seq)
            # print()
            pass
    return dialogue_acts

def equal_da_seq(dialogue_acts, das_seq):
    predict_dialogue_acts = deserialize_dialogue_acts(das_seq)
    for da_type in ['binary', 'categorical', 'non-categorical']:
        das = sorted([(da['intent'], da['domain'], da['slot'], da.get('value', '')) for da in dialogue_acts[da_type]])
        predict_das = sorted([(da['intent'], da['domain'], da['slot'], da.get('value', '')) for da in predict_dialogue_acts[da_type]])
        if das != predict_das:
            return False
    return True
