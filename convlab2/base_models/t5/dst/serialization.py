def serialize_dialogue_state(state):
    state_dict = {}
    for domain in state:
        for slot, value in sorted(state[domain].items()):
            if len(value) > 0:
                state_dict.setdefault(f'[{domain}]', [])
                state_dict[f'[{domain}]'].append(f'[{slot}][{value}]')
    return ';'.join([domain+'('+','.join(slot_values)+')' for domain, slot_values in state_dict.items()])

def deserialize_dialogue_state(state_seq):
    state = {}
    if len(state_seq) == 0:
        return state
    state_seqs = state_seq.split(']);[')  # will consume "])" and "["
    for i, state_seq in enumerate(state_seqs):
        if len(state_seq) == 0 or len(state_seq.split(']([')) != 2:
            continue
        if i == 0:
            if state_seq[0] == '[':
                state_seq = state_seq[1:]
        if i == len(state_seqs) - 1:
            if state_seq[-2:] == '])':
                state_seq = state_seq[:-2]
        
        try:
            domain, slot_values = state_seq.split(']([')
        except:
            continue
        for slot_value in slot_values.split('],['):
            try:
                slot, value = slot_value.split('][')
            except:
                continue
            state.setdefault(domain, {})
            state[domain][slot] = value
    return state

def equal_state_seq(state, state_seq):
    predict_state = deserialize_dialogue_state(state_seq)
    svs = sorted([(domain, slot, value) for domain in state for slot, value in state[domain].items() if len(value)>0])
    predict_svs = sorted([(domain, slot, value) for domain in predict_state for slot, value in predict_state[domain].items() if len(value)>0])
    if svs != predict_svs:
        return False
    return True
