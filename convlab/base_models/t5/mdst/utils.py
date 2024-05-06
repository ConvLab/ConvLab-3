import re
import json
import pandas as pd
import random
from copy import deepcopy


def get_state_update(prev_state, cur_state):
    # T_t = B_t - B_{t-1}
    state = deepcopy(cur_state)
    for domain in prev_state:
        state.setdefault(domain, {})
        for slot in prev_state[domain]:
            if slot not in state[domain]:
                # deletion
                state[domain][slot] = ''
            elif prev_state[domain][slot] == state[domain][slot]:
                # carry-over
                state[domain].pop(slot)
        if len(state[domain]) == 0:
            state.pop(domain)
    return state

def update_state(prev_state, state_update):
    # B_{t-1} + T_t = B_t
    state = deepcopy(prev_state)
    for domain in state_update:
        state.setdefault(domain, {})
        for slot in state_update[domain]:
            state[domain][slot] = state_update[domain][slot]
            if state[domain][slot] == "":
                state[domain].pop(slot)
        if len(state[domain]) == 0:
            state.pop(domain)
    return state

def filter_by_full_state(state, full_state):
    # filter out invalid domain-slot key according to ontology
    for domain in list(state.keys()):
        if domain not in full_state:
            state.pop(domain)
            continue
        for slot in list(state[domain].keys()):
            if slot not in full_state[domain]:
                state[domain].pop(slot)
        if len(state[domain]) == 0:
            state.pop(domain)
    return state

def find_and_replace_value_in_sen(value, sen, tgt_value=None):
    '''
    find value in the sentence, and replace it with tgt_value.
    :param value: str, original value
    :param sen: str, sentence to replace
    :param tgt_value: str, target value.
    :return: if tgt_value is None, return whether the original value is found. Else return replaced sentence if find original value, else None
    '''
    digit2word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
    }
    assert isinstance(value, str)
    pw = '(^|[\s,\.:\?!-])(?P<v>{})([\s,\.:\?!-]|$)'
    pn = '(^|[\s\?!-]|\D[,\.:])(?P<v>{})($|[\s\?!-]|[,\.:]\D|[,\.:]$)'
    if value.isdigit():
        pattern = pn
    else:
        pattern = pw
    p = re.compile(pattern.format(re.escape(value)), re.I)
    m = re.search(p, sen)

    replaced_utt = None
    if m:
        if tgt_value is None:
            return True
        replaced_utt = re.sub(p, lambda matched: matched.group(1)+tgt_value+matched.group(3), sen)
    elif value.isdigit() and value in digit2word:
        value = digit2word[value]
        p = re.compile(pw.format(re.escape(value)), re.I)
        m = re.search(p, sen)
        if m:
            if tgt_value is None:
                return True
            replaced_utt = re.sub(p, lambda matched: matched.group(1)+tgt_value+matched.group(3), sen)
    if tgt_value is None:
        return False
    return replaced_utt

def state2triplets(state, full_state):
    state = filter_by_full_state(state, full_state)
    triplets = set()
    for domain in state:
        for slot in state[domain]:
            triplets.add((domain,slot,' '.join(state[domain][slot].split()).lower()))
    return triplets

def eval_state(pred_state, gold_state, pred_state_update, gold_state_update, full_state):
    # turn state into a Set of (domain, slot, value) triplets
    pred_state_update = state2triplets(pred_state_update, full_state)
    gold_state_update = state2triplets(gold_state_update, full_state)
    pred_state = state2triplets(pred_state, full_state)
    gold_state = state2triplets(gold_state, full_state)
    metrics = {}
    # JGA: joint goal accuracy
    metrics['JGA'] = (pred_state == gold_state)
    # RSA: relative slot accuracy
    all_slots = set()
    correct_slots = 0
    for dsv in gold_state:
        all_slots.add((dsv[0],dsv[1]))
        if dsv in pred_state:
            correct_slots += 1
    for dsv in pred_state:
        all_slots.add((dsv[0],dsv[1]))
    all_slots = len(all_slots)
    if all_slots == 0:
        metrics['RSA'] = None
    else:
        metrics['RSA'] = correct_slots/all_slots
    # TA: turn accuracy
    metrics['TA'] = (pred_state_update.issubset(gold_state) and gold_state_update.issubset(pred_state))
    return metrics

def evaluate_and_write_dial(dial, full_state):
    turns = dial['turns']
    prev_pred_state = {}
    for turn_idx in range(0,len(turns),2):
        turn = turns[turn_idx]
        pred_state = turn['predict_state']
        gold_state = turn['state']
        pred_state_update = get_state_update(prev_pred_state, pred_state)
        gold_state_update = turn['state_update']
        prev_pred_state = pred_state

        turn['predict_state_update'] = pred_state_update
        turn['metrics'] = eval_state(pred_state, gold_state, pred_state_update, gold_state_update, full_state)

        turn['active_domains_metrics'] = {}
        for domain in turn['active_domains']:
            turn['active_domains_metrics'][domain] = eval_state(
                {k:v for k,v in pred_state.items() if k == domain},
                {k:v for k,v in gold_state.items() if k == domain},
                {k:v for k,v in pred_state_update.items() if k == domain},
                {k:v for k,v in gold_state_update.items() if k == domain},
                full_state
            )

        
def label_dial(dial, ignore_active_domain=False):
    all_domains = dial['domains']
    turns = dial['turns']
    prev_gold_state = {}
    prev_active_domains = []
    for turn_idx in range(0,len(turns),2):
        turn = turns[turn_idx]
        gold_state = turn['state']
        gold_state_update = get_state_update(prev_gold_state, gold_state)
        prev_gold_state = gold_state

        turn['state_update'] = gold_state_update
        if not ignore_active_domain:
            if len(gold_state_update) > 0:
                turn['active_domains'] = list(gold_state_update.keys())
                assert len(turn['active_domains']) > 0
                assert all([d in all_domains for d in turn['active_domains']]), print(turn,all_domains)
                if len(prev_active_domains) == 0:
                    for idx in range(0,turn_idx,2):
                        turns[idx]['active_domains'] = turn['active_domains']
                prev_active_domains = turn['active_domains']
            else:
                turn['active_domains'] = prev_active_domains
        
        # turn['cross_domain'] = False
        # turn['cross_domain_slot_value'] = []
        # value2domain_slot = {}
        # for domain in gold_state:
        #     for slot, value in gold_state[domain].items():
        #         assert len(value) > 0, print(gold_state[domain], turn)
        #         value = value.lower()
        #         if value not in ['yes', 'no', 'dontcare']:
        #             value2domain_slot.setdefault(value, [])
        #             value2domain_slot[value].append((domain, slot))
        # for domain in gold_state_update:
        #     for slot, value in gold_state_update[domain].items():
        #         value = value.lower()
        #         if value in value2domain_slot and len(value2domain_slot[value]) > 1:
        #             # multiple value
        #             for d,s in value2domain_slot[value]:
        #                 if d != domain:
        #                     turn['cross_domain'] = True
        #                     turn['cross_domain_slot_value'].append(str((d,s,value,domain,slot)))
        # turn['delta_c'] = 10
        
        # calculate delta_c
        delta_c = 0
        turn['cross_domain'] = False
        turn['cross_domain_slot_value'] = None
        for domain in gold_state_update:
            for slot in gold_state_update[domain]:
                # find value in previous turns
                value = gold_state_update[domain][slot]
                if value in ['yes', 'no']:
                    # if slot not in ['internet', 'parking']:
                    #     print(domain, slot)
                    continue
                value_in_utt = False
                value_cross_domain = True
                for idx in range(turn_idx,-1,-1):
                    if find_and_replace_value_in_sen(value, turns[idx]['utterance']):
                        if not value_in_utt:
                            # nearest turn for delta_c
                            delta_c = max(delta_c, turn_idx-idx)
                            value_in_utt = True
                        if ('active_domains' in turns[idx] and domain in turns[idx]['active_domains']) \
                            or ('active_domains' not in turns[idx] and domain in turns[idx+1]['active_domains']):
                            # cross domain if the domain is not active in that turn
                            value_cross_domain = False
                            break
                if value_in_utt and value_cross_domain:
                    # appear in previous turns, and domain is not active in those turn
                    turn['cross_domain'] = True
                    turn['cross_domain_slot_value'] = str((domain, slot, value))
        turn['delta_c'] = delta_c
        
def dials2qadst_samples(dials, ontology, full_state):
    """generate QA samples from dialogs, following UnifiedQA-v2 format"""
    samples = []
    for idx, dial in enumerate(dials):
        domains = dial['domains']
        history = []
        for turn_idx, turn in enumerate(dial['turns']):
            history.append([turn['speaker'],turn['utterance']])
            if turn['speaker'] != 'user':
                continue
            context = '\n'.join([f"{speaker}: {utt}" for speaker, utt in history])
            
            for domain in domains:
                for slot in full_state[domain]:
                    slot_desc = ontology['domains'][domain]['slots'][slot]['description']
                    if domain not in turn['state'] or slot not in turn['state'][domain]:
                        value = '<no answer>'
                    else:
                        value = turn['state'][domain][slot]

                    samples.append(json.dumps({
                        'domains': dial['domains'], 
                        'dial_idx': idx,
                        'turn_idx': turn_idx,
                        'input': f'what is the {slot_desc} of the {domain.split("_")[0]} domain? \\n {context}'.lower(), 
                        'output': value.lower(),
                        'slot_ori': slot}, ensure_ascii=False)+'\n')
            
    return samples

def dials2qadst_cross_domain_samples(dials, ontology, full_state):
    """generate QA samples from dialogs, following unifiedQA-v2 format. The question is from all domains"""
    samples = []
    for idx, dial in enumerate(dials):
        domains = dial['domains']
        assert len(domains) == 1
        domain = domains[0]
        history = []
        for turn_idx, turn in enumerate(dial['turns']):
            history.append([turn['speaker'],turn['utterance']])
            if turn['speaker'] != 'user':
                continue
            state = turn['state']
        
        context = '\n'.join([f"{speaker}: {utt}" for speaker, utt in history])
            
        for cross_domain in full_state:
            for slot in full_state[cross_domain]:
                slot_desc = ontology['domains'][cross_domain]['slots'][slot]['description']
                if cross_domain != domain or domain not in state or slot not in state[domain]:
                    value = '<no answer>'
                else:
                    value = state[domain][slot]

                samples.append(json.dumps({
                    'domains': [domain, cross_domain], 
                    'dial_idx': idx,
                    'turn_idx': turn_idx,
                    'input': f'what is the {slot_desc} of the {cross_domain.split("_")[0]} domain? \\n {context}'.lower(), 
                    'output': value.lower(),
                    'slot': slot}, ensure_ascii=False)+'\n')
            
    return samples


def dials2domain_cls_samples(dials, context_window_size):
    samples = []
    for idx, dial in enumerate(dials):
        history = []
        for turn_idx, turn in enumerate(dial['turns']):
            history.append([turn['speaker'],turn['utterance']])
            if turn['speaker'] != 'user':
                continue
        
            # context = '\n'.join([f"{speaker}: {utt}" for speaker, utt in history])
            context = '\n'.join([f"[{speaker}] {utt}" for speaker, utt in history[-context_window_size-1:]])
                
            samples.append(json.dumps({
                'domains': dial['domains'],
                'active_domains': turn['active_domains'],
                'dial_idx': idx,
                'turn_idx': turn_idx,
                'input': context,
                'output': ';'.join(turn['active_domains'])}, ensure_ascii=False)+'\n')
            
    return samples

"""
compare_strings is written by OpenAI code-davinci-002 given the following instruction and only add indexes of ops in `difference`
"write a function to compare two strings and tell the differences (insertion, deletion, substitution)"
"""
def compare_strings(str1, str2):
    # Create a matrix to store the results of inner loop
    matrix = [[0 for x in range(len(str2)+1)] for x in range(len(str1)+1)] 
    for i in range(len(str1) + 1): 
        for j in range(len(str2) + 1): 
            if i == 0: 
                matrix[i][j] = j    # Top row is insertion
            elif j == 0: 
                matrix[i][j] = i    # Left row is deletion
            elif str1[i-1] == str2[j-1]: 
                matrix[i][j] = matrix[i-1][j-1] 
            else: 
                matrix[i][j] = 1 + min(matrix[i][j-1],  # Insertion
                                       matrix[i-1][j],  # Deletion
                                       matrix[i-1][j-1]) # Substitution
    # Traverse the matrix to find the differences
    i = len(str1) 
    j = len(str2)  
    differences = []
    while i > 0 and j > 0: 
        if str1[i-1] == str2[j-1]: 
            i -= 1
            j -= 1
        elif matrix[i][j] == matrix[i-1][j-1] + 1: 
            differences.append(("Substitution", str1[i-1], str2[j-1], i-1, j-1))
            i -= 1
            j -= 1
        elif matrix[i][j] == matrix[i-1][j] + 1: 
            differences.append(("Deletion", str1[i-1], None, i-1, j)) 
            i -= 1
        else: 
            differences.append(("Insertion", None, str2[j-1], i, j-1)) 
            j -= 1
    # Add remaining deletions
    while i > 0: 
        differences.append(("Deletion", str1[i-1], None, i-1, j)) 
        i -= 1
    # Add remaining insertions
    while j > 0:
        differences.append(("Insertion", None, str2[j-1], i, j-1))
        j -= 1
    return differences[::-1]

def label_str(src_str, dst_str, substitution="sub", deletion="del", insertion="ins"):
    """label the Substitution, Deletion, and insertion from src_str to dst_str
    example:
    src_str: "Why does not the Spiderman Peter Parker kill the Octopus?"
    dst_str: "Why doesn't the Spiderman kill the Dr. Octopus?"
    return: "Why <sub>does not</sub> the Spiderman <del>Peter Parker</del> kill the <ins></ins> Octopus?"
    """
    src_str = src_str.split()
    dst_str = dst_str.split()
    differences = compare_strings(src_str, dst_str)
    src_labels = ['O'] * (len(src_str)+1)
    for difference in differences:
        label = difference[0][0]
        idx = difference[3]
        src_labels[idx] = label
    op_spans = []
    i = 0
    while i < len(src_labels):
        label = src_labels[i]
        if label != 'O':
            span_label = label
            j = i+1
            while j < len(src_labels) and src_labels[j] != 'O':
                if src_labels[j] == 'S':
                    span_label = 'S'
                j += 1
            op_spans.append([span_label, i, j-1])
            i = j
        i += 1
    
    for op, start, end in op_spans:
        if op == 'S':
            src_str[start] = f'<{substitution}> '+src_str[start]
            src_str[end] = src_str[end]+f' </{substitution}>'
        elif op == 'D':
            src_str[start] = f'<{deletion}> '+src_str[start]
            src_str[end] = src_str[end]+f' </{deletion}>'
        else:
            for idx in range(start,end+1):
                if idx >= len(src_str):
                    src_str.append(f'<{insertion}></{insertion}>')
                else:
                    src_str[idx] = f'<{insertion}></{insertion}> '+src_str[idx]

    return ' '.join(src_str)

def merge_single_dials(dials):
    # merge multiple single domain dials into one multi-domain dials
    random.shuffle(dials)
    new_dial = {'domains': deepcopy(dials[0]['domains']), 'turns':deepcopy(dials[0]['turns'])}
    new_state = deepcopy(new_dial['turns'][-2]['state'])
    for dial in dials[1:]:
        # delete last two turns (ending) of previous dialog if no state update
        if len(new_dial['turns'][-2]['state_update']) == 0:
            del new_dial['turns'][-2:]
            new_state = deepcopy(new_dial['turns'][-2]['state'])
        domains = dial['domains']
        assert len(domains) == 1, print(domains)
        domain = domains[0]
        new_dial['domains'].append(domain)
        for turn in dial['turns']:
            new_dial['turns'].append(deepcopy(turn))
            # accumulate turn state
            if turn['speaker'] == 'user':
                for domain in turn['state']:
                    new_state[domain] = turn['state'][domain]
                new_dial['turns'][-1]['state'] = deepcopy(new_state)
    return new_dial

def merge_single_domain_rel_dials(dial1, dial2):
    # merge two dials (`(train|validation)_single_domain_qa.json`) that have at least one slot-pair shares same value
    new_dial = deepcopy(dial1)
    if len(new_dial['turns'][-2]['state_update']) == 0:
        del new_dial['turns'][-2:]
    new_dial.pop('qa')
    new_dial.pop('state')
    domain1, domain2 = dial1['domains'][0], dial2['domains'][0]
    state1 = dial1['state'][domain1]
    state2 = dial2['state'][domain2]
    # find slots in domain2 that can have the same value as the slots in domain1
    coref_slots = {}
    for slot2, (slot1, value1) in dial1['qa'][domain2].items():
        if slot2 not in state2:
            continue
        assert value1.lower() == state1[slot1].lower(), print(state1, slot1, dial1['qa'][domain2])
        coref_slots[slot2] = (slot1, state1[slot1])
    assert len(coref_slots) > 0, print(state1, dial1['qa'][domain2], state2)
    new_dial['domains'] = [domain1, domain2]
    new_dial['switch_turn'] = len(new_dial['turns'])

    # append dial2 turns
    for turn in dial2['turns']:
        new_turn = deepcopy(turn)
        if 'state' in turn:
            new_turn.pop('state')
        new_dial['turns'].append(new_turn)

    # modified the merge dialog by replacing tgt_slot_value with src_slot_value
    for tgt_slot, (src_slot, src_slot_value) in coref_slots.items():
        tgt_slot_appear_turn_idxes = []
        tgt_slot_value = None
        for turn_idx in range(new_dial['switch_turn'], len(new_dial['turns'])):
            turn = new_dial['turns'][turn_idx]
            # find the turn that target slot first appears
            if 'state_update' in turn and domain2 in turn['state_update'] and tgt_slot in turn['state_update'][domain2]:
                tgt_slot_appear_turn_idxes.append(turn_idx)
                if len(tgt_slot_appear_turn_idxes) == 1:
                    # first time, change state_update
                    tgt_slot_value = turn['state_update'][domain2][tgt_slot]
                    turn.setdefault('replace_state', [])
                    turn['replace_state'].append({'target':[domain2, tgt_slot, tgt_slot_value], 
                        'source': [domain1, src_slot, src_slot_value]})
                    turn['state_update'][domain2][tgt_slot] = src_slot_value
                else:
                    # second time
                    break
        assert tgt_slot_value is not None, print(tgt_slot, coref_slots[tgt_slot], state2)
        start_turn= max(new_dial['switch_turn'], tgt_slot_appear_turn_idxes[0]-1)
        if len(tgt_slot_appear_turn_idxes) > 1:
            end_turn = tgt_slot_appear_turn_idxes[1]
        else:
            end_turn = len(new_dial['turns']) - 1
        for turn_idx in range(start_turn, end_turn+1):
            new_dial['turns'][turn_idx].setdefault('value2replace', [])
            new_dial['turns'][turn_idx]['value2replace'].append({'source': src_slot_value, 'target': tgt_slot_value})

    # replace values in utterance
    for turn_idx in range(new_dial['switch_turn'], len(new_dial['turns'])):
        if 'value2replace' not in new_dial['turns'][turn_idx]:
            continue
        value2replace = new_dial['turns'][turn_idx].pop('value2replace')
        utt4rewrite = new_dial['turns'][turn_idx]['utterance']
        value_map = {}
        for i, slot_pair in enumerate(value2replace):
            tgt_slot_value, src_slot_value = slot_pair['target'], slot_pair['source']
            value_place_holder = f'__@{i}@__'
            coref_utt = find_and_replace_value_in_sen(tgt_slot_value, utt4rewrite, value_place_holder)
            if coref_utt is not None:
                utt4rewrite = coref_utt
                value_map[value_place_holder] = f'<sub> {src_slot_value} </sub>'
        if len(value_map) == 0:
            continue
        for ph, v in value_map.items():
            utt4rewrite = utt4rewrite.replace(ph, v)
        new_dial['turns'][turn_idx]['original_utt'] = new_dial['turns'][turn_idx]['utterance']
        new_dial['turns'][turn_idx]['utterance4rewrite'] = utt4rewrite
        new_dial['turns'][turn_idx]['utterance'] = utt4rewrite.replace('<sub> ','').replace(' </sub>', '')

    for turn_idx in range(new_dial['switch_turn'], len(new_dial['turns'])):
        turn = new_dial['turns'][turn_idx]
        if 'state_update' in turn:
            turn['state'] = update_state(new_dial['turns'][turn_idx-2]['state'], turn['state_update'])

    new_dial['switch_turn'] = [new_dial['switch_turn']]
    label_dial(new_dial, ignore_active_domain=True)
    return new_dial

def mergeN_single_domain_rel_dials(dials):
    # merge N dials
    dials = [deepcopy(dial) for dial in dials]
    domains = []
    states = {}
    for dial_idx, dial in enumerate(dials):
        if dial_idx + 1 < len(dials):
            # not last dial, remove the ending turn
            if len(dial['turns'][-2]['state_update']) == 0:
                del dial['turns'][-2:]
        assert len(dial['domains']) == 1
        domain = dial['domains'][0]
        domains.append(domain)
        assert domain not in states and domain in dial['state'] and len(dial['state']) == 1
        states[domain] = dial['state'][domain]

        if dial_idx == 0:
            continue

        coref_slots = {}
        for prev_dial in dials[:dial_idx]:
            if 'qa' not in prev_dial or domain not in prev_dial['qa']:
                continue
            for dst_slot, (src_slot, _) in prev_dial['qa'][domain].items():
                if dst_slot not in states[domain]:
                    # non-empty target slot
                    continue
                src_domain = prev_dial['domains'][0]
                # assert src_slot in states[src_domain], print(states, src_domain, src_slot, domain, dst_slot)
                # assert src_value.lower() == states[src_domain][src_slot].lower(), print(states, src_domain, src_slot, domain, dst_slot)
                coref_slots.setdefault(dst_slot, [])
                # could have multiple reference choices
                coref_slots[dst_slot].append((src_domain, src_slot, states[src_domain][src_slot]))
        assert len(coref_slots) > 0, print(len(dials), states, [prev_dial['qa'] for prev_dial in dials[:dial_idx] if 'qa' in prev_dial])
        # random select between multiple reference if the value are not the same
        for dst_slot in coref_slots:
            src_value2idx = {}
            for idx, (src_domain, src_slot, src_value) in enumerate(coref_slots[dst_slot]):
                # if there are multiple same value of different domain, only consider the latest domain
                src_value2idx[src_value] = idx
            idx = random.choice(list(src_value2idx.values()))
            coref_slots[dst_slot] = coref_slots[dst_slot][idx]

        for turn in dial['turns']:
            if 'state' in turn:
                turn.pop('state')

        for dst_slot, (src_domain, src_slot, src_value) in coref_slots.items():
            dst_slot_appear_turn_idxes = []
            dst_slot_value = None
            for turn_idx, turn in enumerate(dial['turns']):
                # find the turn that target slot first appears
                if 'state_update' in turn and domain in turn['state_update'] and dst_slot in turn['state_update'][domain]:
                    dst_slot_appear_turn_idxes.append(turn_idx)
                    if len(dst_slot_appear_turn_idxes) == 1:
                        # first time, change state_update
                        dst_slot_value = turn['state_update'][domain][dst_slot]
                        turn.setdefault('replace_state', [])
                        turn['replace_state'].append({'target':[domain, dst_slot, dst_slot_value], 
                                                      'source': [src_domain, src_slot, src_value]})
                        turn['state_update'][domain][dst_slot] = src_value
                    else:
                        break

            assert dst_slot_value is not None, print(dst_slot, coref_slots[dst_slot], states[domain])
            start_turn = max(0, dst_slot_appear_turn_idxes[0]-1)
            if len(dst_slot_appear_turn_idxes) > 1:
                end_turn = dst_slot_appear_turn_idxes[1]
            else:
                end_turn = len(dial['turns']) - 1
            for turn_idx in range(start_turn, end_turn+1):
                dial['turns'][turn_idx].setdefault('value2replace', [])
                dial['turns'][turn_idx]['value2replace'].append({'source': src_value, 'target': dst_slot_value})
        
        for turn_idx, turn in enumerate(dial['turns']):
            if 'value2replace' not in turn:
                continue
            value2replace = turn.pop('value2replace')
            utt4rewrite = turn['utterance']
            value_map = {}
            for i, slot_pair in enumerate(value2replace):
                dst_value, src_value = slot_pair['target'], slot_pair['source']
                value_place_holder = f'__@{i}@__'
                coref_utt = find_and_replace_value_in_sen(dst_value, utt4rewrite, value_place_holder)
                if coref_utt is not None:
                    utt4rewrite = coref_utt
                    value_map[value_place_holder] = f'<sub> {src_value} </sub>'
            if len(value_map) == 0:
                continue
            for ph, v in value_map.items():
                utt4rewrite = utt4rewrite.replace(ph, v)
            turn['original_utt'] = turn['utterance']
            turn['utterance4rewrite'] = utt4rewrite
            turn['utterance'] = utt4rewrite.replace('<sub> ','').replace(' </sub>', '')

        for turn_idx, turn in enumerate(dial['turns']):
            if 'state_update' in turn:
                if turn_idx - 2 < 0:
                    turn['state'] = turn['state_update']
                else:
                    turn['state'] = update_state(dial['turns'][turn_idx-2]['state'], turn['state_update'])

        for turn_idx, turn in enumerate(dial['turns']):
            if 'replace_state' in turn and 'utterance4rewrite' in turn:
                # only modify user utt
                dial['target_turn_idx'] = turn_idx
                break

        # update states for subsequence dialogs
        states[domain] = dial['turns'][-2]['state'][domain]

    # now concat all dials
    new_dial = dials[0]
    del new_dial['qa']
    del new_dial['state']
    new_dial['domains'] = domains
    new_dial['switch_turn'] = []
    new_dial['target_turn_idx'] = []

    for dial in dials[1:]:
        new_dial['switch_turn'].append(len(new_dial['turns']))
        if 'target_turn_idx' in dial:
            new_dial['target_turn_idx'].append(len(new_dial['turns'])+dial['target_turn_idx'])
        for turn in dial['turns']:
            new_dial['turns'].append(turn)
            if 'state' in turn:
                new_dial['turns'][-1]['state'] = update_state(new_dial['turns'][-3]['state'], turn['state_update'])
    
    label_dial(new_dial, ignore_active_domain=True)
    return new_dial

def get_value_from_bio_tags(sentence, entity_groups):
    entities = []
    if len(entity_groups) == 0:
        return entities
    for entity in entity_groups:
        tag, token, start, end = entity['entity_group'], entity['word'], entity['start'], entity['end']
        if tag == 'B':
            entities.append(entity)
        else:
            assert tag == 'I'
            if len(entities) != 0 and start - entities[-1]['end'] <= 1:
                entities[-1]['end'] = end
            elif len(entities) == 0 or start - entities[-1]['end'] > 1:
                entities.append(entity)

    return [sentence[x['start']:x['end']] for x in entities]

def value_in_value_list(value, value_list):
    return any([value in v or v in value for v in value_list])

def filter_rewrite(ori_sen, gen_sen, values2rm, ori_values, gen_values, recover_sen=None):
    values2rm = [value.strip().lower() for value in values2rm]
    ori_values = [value.strip().lower() for value in ori_values]
    gen_values = [value.strip().lower() for value in gen_values]
    for value in values2rm:
        if value in gen_sen.lower():
            # if the src_value still in gen_sen
            return False
        if recover_sen is not None and value not in recover_sen.lower():
            return False
    for value in ori_values:
        if any([value in v for v in values2rm]):
            # if ori_value is part of src_value, it should not appear in gen_values
            if value_in_value_list(value, gen_values):
                # if ori_value appear in gen_values
                return False
        else:
            # if ori_value is not src_value, it should appear in gen_values
            if not value_in_value_list(value, gen_values):
                return False
    for value in gen_values:
        # if gen_value not in ori_values, it is redundant
        if not value_in_value_list(value, ori_values):
            return False
    return True
