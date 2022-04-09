def serialize_dialogue_state(state):
    state_seqs = []
    for domain in state:
        for slot, value in state[domain].items():
            if len(value) > 0:
                state_seqs.append(f'[{domain}][{slot}][{value}]')
    
    return ';'.join(state_seqs)

def deserialize_dialogue_state(state_seq):
    state = {}
    if len(state_seq) == 0:
        return state
    state_seqs = state_seq.split('];[')
    for i, state_seq in enumerate(state_seqs):
        if len(state_seq) == 0:
            continue
        if i == 0:
            if state_seq[0] == '[':
                state_seq = state_seq[1:]
        if i == len(state_seqs) - 1:
            if state_seq[-1] == ']':
                state_seq = state_seq[:-1]
        s = state_seq.split('][')
        if len(s) != 3:
            continue
        domain, slot, value = s
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
