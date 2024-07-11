"""process data according to different models & data augmentation methods"""
import json
import random
import pandas as pd
from copy import deepcopy
from pprint import pprint
import re
from convlab.base_models.t5.dst.serialization import serialize_dialogue_state, deserialize_dialogue_state
from convlab.base_models.t5.mdst.utils import get_state_update, update_state, filter_by_full_state, merge_single_dials, merge_single_domain_rel_dials, mergeN_single_domain_rel_dials


class DataProcessor:
    AUG_TYPE_NONE=0
    AUG_TYPE_REPLACE_TRUE=1 
    AUG_TYPE_CONCAT2=2
    AUG_TYPE_CONCATN=3
    AUG_TYPE_CONCAT2REL=4
    AUG_TYPE_CONCAT2ANA=5
    AUG_TYPE_CONCAT2ELL=6
    AUG_TYPE_CONCAT2MIX=7
    AUG_TYPE_CONCATNREL=8
    AUG_TYPE_CONCATNMIX=9
    AUG_TYPE_CONCAT2GPT=10
    AUG_TYPE_CONCATNANA=11
    AUG_TYPE_10TRUE_AND_SYN=12
    AUG_TYPE_15TRUE_AND_SYN=13
    AUG_TYPE_10TRUE=14
    AUG_TYPE_15TRUE=15
    AUG_TYPE_TRUE_AND_SYN=16
    AUG_TYPE_CONCAT2ANA_UNIFORM=17
    AUG_TYPE_10SYN=18
    AUG_TYPE_15SYN=19
    MODEL_TYPE_T5DST=0
    MODEL_TYPE_MinTL=1
    MODEL_TYPE_SDPDST=3
    MODEL_TYPE_HT5DST=4
    MODEL_TYPE_HMinTL=5
    MODEL_TYPE_HSDPDST=7
    
    def __init__(self, model_type, data_aug_type, context_window_size, ontology, full_state):
        """
        model_type: 
        0: Context => full state B_t
        1: B_{t-1}; ... S_{t-2},U_{t-1},S_{t-1},U_t => L_t (B_t-B_{t-1}). Predict state update
        2: B_{t-2}; ... S_{t-2},U_{t-1},S_{t-1},U_t => L_{t-1} || L_t. Review last turn state update
        """
        self.model_type = model_type
        self.data_aug_type = data_aug_type
        self.context_window_size = context_window_size
        self.ontology = ontology
        self.full_state = full_state

    def dials2samples(self, dials):
        """generate DST samples from dialogs"""
        samples = []
        for dial_idx, dial in enumerate(dials):
            history = []
            turns_active_domains = []
            for turn_idx, turn in enumerate(dial['turns']):
                history.append([turn['speaker'],turn['utterance']])
                if 'active_domains' in turn:
                    turns_active_domains.append(turn['active_domains'])
                else:
                    continue
                if self.model_type in [DataProcessor.MODEL_TYPE_HT5DST, DataProcessor.MODEL_TYPE_HMinTL, DataProcessor.MODEL_TYPE_HSDPDST]:
                    active_domains = turn['active_domains']
                    start_turn_idx = 0
                    for idx, turn_domains in enumerate(turns_active_domains):
                        if any([domain in turn_domains for domain in active_domains]):
                            start_turn_idx = idx+idx
                            break
                    context = '\n'.join([f"[{speaker}] {utt}" for speaker, utt in history[start_turn_idx:]])

                    prev_domains_state = {k:v for k,v in turn['state'].items() if k not in active_domains}
                    prev_domains_state_seq = serialize_dialogue_state(prev_domains_state)

                    active_domains_seq = ";".join([f'[{d}]' for d in active_domains])
                else:
                    context = '\n'.join([f"[{speaker}] {utt}" for speaker, utt in history[-self.context_window_size-1:]])
                
                if self.model_type == DataProcessor.MODEL_TYPE_T5DST:
                    # T5DST
                    state_seq = serialize_dialogue_state(turn['state'])
                    samples.append(json.dumps({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': context, 
                        'output': state_seq}, ensure_ascii=False)+'\n')

                elif self.model_type == DataProcessor.MODEL_TYPE_MinTL:
                    # MinTL
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    prev_state_seq = serialize_dialogue_state(prev_state)
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update, ignore_empty_value=False)
                    samples.append(json.dumps({
                        'domains': dial['domains'],
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{context}\n\n{prev_state_seq}', 
                        'output': state_update_seq}, ensure_ascii=False)+'\n')

                elif self.model_type == DataProcessor.MODEL_TYPE_SDPDST:
                    # SDP-DST
                    for domain in self.full_state:
                        domain_desc = self.ontology['domains'][domain]['description']
                        for slot in self.full_state[domain]:
                            slot_desc = self.ontology['domains'][domain]['slots'][slot]['description']
                            schema_prompt = f"[domain] {domain} {domain_desc} [slot] {slot} {slot_desc}"
                            if self.ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                possible_values = ", ".join(self.ontology['domains'][domain]['slots'][slot]["possible_values"])
                                schema_prompt += f" [PVs] {possible_values}"
                            
                            if domain not in turn['state'] or slot not in turn['state'][domain]:
                                value = 'NONE'
                            else:
                                value = turn['state'][domain][slot]

                            samples.append(json.dumps({
                                'domains': dial['domains'], 
                                'dial_idx': dial_idx,
                                'turn_idx': turn_idx,
                                'input': f'{context}\n\n{schema_prompt}', 
                                'output': value,
                                'domain': domain,
                                'slot': slot}, ensure_ascii=False)+'\n')
                
                elif self.model_type == DataProcessor.MODEL_TYPE_HT5DST:
                    cur_domains_state = {k:v for k,v in turn['state'].items() if k in active_domains}
                    cur_domains_state_seq = serialize_dialogue_state(cur_domains_state)
                    samples.append(json.dumps({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{prev_domains_state_seq}\n\n{context}\n\n{active_domains_seq}'.strip(),
                        'output': cur_domains_state_seq}, ensure_ascii=False)+'\n')
                    
                elif self.model_type == DataProcessor.MODEL_TYPE_HMinTL:
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update, ignore_empty_value=False)

                    for domain in state_update:
                        assert domain in active_domains
                    cur_domains_prev_state = {k:v for k,v in prev_state.items() if k in active_domains}
                    prev_state_seq = serialize_dialogue_state(cur_domains_prev_state)
                    
                    samples.append(json.dumps({
                        'domains': dial['domains'],
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{prev_domains_state_seq}\n\n{context}\n\n{prev_state_seq}\n\n{active_domains_seq}'.strip(), 
                        'output': state_update_seq}, ensure_ascii=False)+'\n')
                    
                elif self.model_type == DataProcessor.MODEL_TYPE_HSDPDST:
                    for domain in active_domains:
                        # only active domains
                        domain_desc = self.ontology['domains'][domain]['description']
                        for slot in self.full_state[domain]:
                            slot_desc = self.ontology['domains'][domain]['slots'][slot]['description']
                            schema_prompt = f"[domain] {domain} {domain_desc} [slot] {slot} {slot_desc}"
                            if self.ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                possible_values = ", ".join(self.ontology['domains'][domain]['slots'][slot]["possible_values"])
                                schema_prompt += f" [PVs] {possible_values}"
                            
                            if domain not in turn['state'] or slot not in turn['state'][domain]:
                                value = 'NONE'
                            else:
                                value = turn['state'][domain][slot]

                            samples.append(json.dumps({
                                'domains': dial['domains'], 
                                'dial_idx': dial_idx,
                                'turn_idx': turn_idx,
                                'input': f'{prev_domains_state_seq}\n\n{context}\n\n{schema_prompt}'.strip(), 
                                'output': value,
                                'domain': domain,
                                'slot': slot}, ensure_ascii=False)+'\n')
                
                elif self.model_type == 2:
                    prev_prev_state = {} if turn_idx < 4 else dial['turns'][turn_idx-4]['state']
                    prev_prev_state_seq = serialize_dialogue_state(prev_prev_state)
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    prev_state_update = get_state_update(prev_prev_state, prev_state)
                    prev_state_update_seq = serialize_dialogue_state(prev_state_update)
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update)
                    samples.append(json.dumps({
                        'domains': dial['domains'],
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'Previous state: {prev_prev_state_seq}\n\nContext: {context}', 
                        'output': f'{prev_state_update_seq} || {state_update_seq}'}, ensure_ascii=False)+'\n')
                
        return samples

    def read_turns_from_dials(self, dials, read_turn_idx):
        """for turn by turn inference, prepare sample with previous prediction"""
        samples = []
        for dial_idx, dial in enumerate(dials):
            history = []
            turns_active_domains = []
            for turn_idx, turn in enumerate(dial['turns']):
                history.append([turn['speaker'],turn['utterance']])
                if 'predict_active_domains' in turn:
                    turns_active_domains.append(turn['predict_active_domains'])
                if turn_idx != read_turn_idx:
                    continue
                if self.model_type in [DataProcessor.MODEL_TYPE_HT5DST, DataProcessor.MODEL_TYPE_HMinTL, DataProcessor.MODEL_TYPE_HSDPDST]:
                    active_domains = turn['predict_active_domains']
                    start_turn_idx = 0
                    for idx, turn_domains in enumerate(turns_active_domains):
                        if any([domain in turn_domains for domain in active_domains]):
                            start_turn_idx = idx+idx
                            break
                    context = '\n'.join([f"[{speaker}] {utt}" for speaker, utt in history[start_turn_idx:]])

                    prev_predict_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['predict_state']
                    prev_domains_state = {k:v for k,v in prev_predict_state.items() if k not in active_domains}
                    prev_domains_state_seq = serialize_dialogue_state(prev_domains_state)

                    active_domains_seq = ";".join([f'[{d}]' for d in active_domains])
                else:
                    context = '\n'.join([f"[{speaker}] {utt}" for speaker, utt in history[-self.context_window_size-1:]])
                
                if self.model_type == DataProcessor.MODEL_TYPE_T5DST:
                    state_seq = serialize_dialogue_state(turn['state'])
                    samples.append({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': context, 
                        'output': state_seq})

                elif self.model_type == DataProcessor.MODEL_TYPE_MinTL:
                    prev_predict_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['predict_state']
                    prev_predict_state_seq = serialize_dialogue_state(prev_predict_state)
                    
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update, ignore_empty_value=False)
                    samples.append({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{context}\n\n{prev_predict_state_seq}', 
                        'output': state_update_seq})
                    
                elif self.model_type == DataProcessor.MODEL_TYPE_SDPDST:
                    # DONE
                    for domain in self.full_state:
                        domain_desc = self.ontology['domains'][domain]['description']
                        for slot in self.full_state[domain]:
                            slot_desc = self.ontology['domains'][domain]['slots'][slot]['description']
                            schema_prompt = f"[domain] {domain} {domain_desc} [slot] {slot} {slot_desc}"
                            if self.ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                possible_values = ", ".join(self.ontology['domains'][domain]['slots'][slot]["possible_values"])
                                schema_prompt += f" [PVs] {possible_values}"
                            
                            if domain not in turn['state'] or slot not in turn['state'][domain]:
                                value = 'NONE'
                            else:
                                value = turn['state'][domain][slot]

                            samples.append({
                                'domains': dial['domains'], 
                                'dial_idx': dial_idx,
                                'turn_idx': turn_idx,
                                'input': f'{context}\n\n{schema_prompt}', 
                                'output': value,
                                'domain': domain,
                                'slot': slot})
                            
                elif self.model_type == DataProcessor.MODEL_TYPE_HT5DST:
                    cur_domains_state = {k:v for k,v in turn['state'].items() if k in turn['active_domains']}
                    cur_domains_state_seq = serialize_dialogue_state(cur_domains_state)
                    samples.append({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{prev_domains_state_seq}\n\n{context}\n\n{active_domains_seq}'.strip(),
                        'output': cur_domains_state_seq})

                elif self.model_type == DataProcessor.MODEL_TYPE_HMinTL:
                    cur_domains_prev_predict_state = {k:v for k,v in prev_predict_state.items() if k in active_domains}
                    prev_predict_state_seq = serialize_dialogue_state(cur_domains_prev_predict_state)
                    
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update, ignore_empty_value=False)
                    
                    samples.append({
                        'domains': dial['domains'],
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'{prev_domains_state_seq}\n\n{context}\n\n{prev_predict_state_seq}\n\n{active_domains_seq}'.strip(), 
                        'output': state_update_seq})

                elif self.model_type == DataProcessor.MODEL_TYPE_HSDPDST:
                    for domain in active_domains:
                        # only active domains
                        if domain not in self.full_state:
                            continue
                        domain_desc = self.ontology['domains'][domain]['description']
                        for slot in self.full_state[domain]:
                            slot_desc = self.ontology['domains'][domain]['slots'][slot]['description']
                            schema_prompt = f"[domain] {domain} {domain_desc} [slot] {slot} {slot_desc}"
                            if self.ontology['domains'][domain]['slots'][slot]['is_categorical']:
                                possible_values = ", ".join(self.ontology['domains'][domain]['slots'][slot]["possible_values"])
                                schema_prompt += f" [PVs] {possible_values}"
                            
                            if domain not in turn['state'] or slot not in turn['state'][domain]:
                                value = 'NONE'
                            else:
                                value = turn['state'][domain][slot]

                            samples.append({
                                'domains': dial['domains'], 
                                'dial_idx': dial_idx,
                                'turn_idx': turn_idx,
                                'input': f'{prev_domains_state_seq}\n\n{context}\n\n{schema_prompt}'.strip(), 
                                'output': value,
                                'domain': domain,
                                'slot': slot})

                elif self.model_type == 2:
                    prev_prev_predict_state = {} if turn_idx < 4 else dial['turns'][turn_idx-4]['predict_state']
                    prev_prev_predict_state_seq = serialize_dialogue_state(prev_prev_predict_state)

                    prev_prev_state = {} if turn_idx < 4 else dial['turns'][turn_idx-4]['state']
                    prev_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['state']
                    prev_state_update = get_state_update(prev_prev_state, prev_state)
                    prev_state_update_seq = serialize_dialogue_state(prev_state_update)
                    state_update = get_state_update(prev_state, turn['state'])
                    state_update_seq = serialize_dialogue_state(state_update)
                    samples.append({
                        'domains': dial['domains'], 
                        'dial_idx': dial_idx,
                        'turn_idx': turn_idx,
                        'input': f'Previous state: {prev_prev_predict_state_seq}\n\nContext: {context}', 
                        'output': f'{prev_state_update_seq} || {state_update_seq}'})
                
                turn['model_input'] = samples[-1]['input']
                break

        return samples

    def write_turns_to_dials(self, dials, write_turn_idx, samples2write):
        """for turn by turn inference, write prediction to original turn['predict_state']"""
        cnt = 0
        turn_idx = write_turn_idx
        for dial in dials:
            if turn_idx < len(dial['turns']):
                turn = dial['turns'][turn_idx]
                prediction = samples2write[cnt]

                if self.model_type in [DataProcessor.MODEL_TYPE_HT5DST, DataProcessor.MODEL_TYPE_HMinTL, DataProcessor.MODEL_TYPE_HSDPDST]:
                    active_domains = turn['predict_active_domains']
                    
                    prev_predict_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['predict_state']
                    prev_domains_state = {k:v for k,v in prev_predict_state.items() if k not in active_domains}
                
                if self.model_type == DataProcessor.MODEL_TYPE_T5DST:
                    turn['prediction'] = prediction
                    turn['predict_state'] = filter_by_full_state(deserialize_dialogue_state(prediction), self.full_state)

                elif self.model_type == DataProcessor.MODEL_TYPE_MinTL:
                    turn['prediction'] = prediction
                    prev_predict_state = {} if turn_idx < 2 else dial['turns'][turn_idx-2]['predict_state']
                    predict_state_update = filter_by_full_state(deserialize_dialogue_state(prediction), self.full_state)
                    predict_state = update_state(prev_predict_state, predict_state_update)
                    turn['predict_state'] = predict_state

                elif self.model_type == DataProcessor.MODEL_TYPE_SDPDST:
                    # TODO
                    predict_state = {}
                    for domain in self.full_state:
                        for slot in self.full_state[domain]:
                            value = samples2write[cnt]
                            if value != 'NONE' and value != '':
                                predict_state.setdefault(domain, {})
                                predict_state[domain][slot] = value
                            cnt += 1
                    cnt -= 1
                    turn['prediction'] = prediction
                    turn['predict_state'] = predict_state

                elif self.model_type == DataProcessor.MODEL_TYPE_HT5DST:
                    turn['prediction'] = prediction
                    cur_domains_state = filter_by_full_state({k:v for k,v in deserialize_dialogue_state(prediction).items() if k in active_domains}, 
                                                             self.full_state)
                    turn['predict_state'] = {**prev_domains_state, **cur_domains_state}

                elif self.model_type == DataProcessor.MODEL_TYPE_HMinTL:
                    turn['prediction'] = prediction
                    predict_state_update = filter_by_full_state({k:v for k,v in deserialize_dialogue_state(prediction).items() if k in active_domains},
                                                                self.full_state)
                    predict_state = update_state(prev_predict_state, predict_state_update)
                    turn['predict_state'] = predict_state

                elif self.model_type == DataProcessor.MODEL_TYPE_HSDPDST:
                    predict_state = {}
                    for domain in active_domains:
                        if domain not in self.full_state:
                            continue
                        for slot in self.full_state[domain]:
                            value = samples2write[cnt]
                            if value != 'NONE' and value != '':
                                predict_state.setdefault(domain, {})
                                predict_state[domain][slot] = value
                            cnt += 1
                    cnt -= 1
                    turn['prediction'] = prediction
                    turn['predict_state'] = {**prev_domains_state, **predict_state}

                elif self.model_type == 2:
                    turn['prediction'] = prediction
                    assert len(prediction.split('||')) == 2
                    p1, p2 = prediction.split('||')
                    prev_prev_predict_state = {} if turn_idx < 4 else dial['turns'][turn_idx-4]['predict_state_refine']
                    predict_prev_state_update = deserialize_dialogue_state(p1)
                    prev_predict_state = update_state(prev_prev_predict_state, predict_prev_state_update)
                    if turn_idx >= 2:
                        dial['turns'][turn_idx-2]['predict_state_refine'] = prev_predict_state

                    predict_state_update = deserialize_dialogue_state(p2)
                    predict_state = update_state(prev_predict_state, predict_state_update)
                    turn['predict_state'] = predict_state
                
                cnt += 1

    def data_augmentation(self, single_domain_dials, multi_domain_dials, num_aug_times, **kwargs):
        random.seed(42)
        single_domain_dial_idxes_by_domain = {}
        for dial_idx, dial in enumerate(single_domain_dials):
            assert len(dial['domains']) == 1
            domain = dial['domains'][0]
            single_domain_dial_idxes_by_domain.setdefault(domain, [])
            single_domain_dial_idxes_by_domain[domain].append(dial_idx)

        aug_dials = []
        num_aug_dials = round(len(single_domain_dials) * num_aug_times)
        print('aug dials:', num_aug_dials)
        if self.data_aug_type == DataProcessor.AUG_TYPE_REPLACE_TRUE:
            # replace single domain dials with true multi-domain dials
            random.shuffle(multi_domain_dials)
            for domain in single_domain_dial_idxes_by_domain:
                random.shuffle(single_domain_dial_idxes_by_domain[domain])
            remove_idxes = []
            i = 0
            while num_aug_dials > 0:
                if i >= len(multi_domain_dials):
                    raise IndexError("not enough true multi-domain dials for replacement")
                dial = multi_domain_dials[i]
                i += 1
                domains = dial['domains']
                # skip when single domain dialogs are consumed
                if any([len(single_domain_dial_idxes_by_domain[domain]) == 0 for domain in domains]):
                    continue
                # cnt a single domain dialog in each domain
                dial['dials_idx'] = []
                for domain in domains:
                    remove_idxes.append(single_domain_dial_idxes_by_domain[domain].pop())
                    dial['dials_idx'].append(remove_idxes[-1])
                aug_dials.append(dial)
                num_aug_dials -= len(domains)
            # remove the single domain dialogs as they are traded for multi-domain dialogs
            for dial_idx in sorted(remove_idxes, reverse=True):
                del single_domain_dials[dial_idx]
        
        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCAT2:
            # concat dials from 2 randomly chosen domains
            all_domains = list(single_domain_dial_idxes_by_domain.keys())
            while num_aug_dials > 0:
                d1, d2 = random.sample(all_domains,2)
                d1_idx, d2_idx = random.choice(single_domain_dial_idxes_by_domain[d1]), random.choice(single_domain_dial_idxes_by_domain[d2])
                dial = merge_single_dials([single_domain_dials[d1_idx], 
                                           single_domain_dials[d2_idx]])
                dial['dials_idx'] = [d1_idx, d2_idx]
                dial['dials_idx'] = [int(idx) for idx in dial['dials_idx']]
                aug_dials.append(dial)
                num_aug_dials -= len(dial['domains'])

        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCATN:
            # with probability p=0.7/0.2/0.1 to append 1/2/3 dial from a new domain
            all_domains = list(single_domain_dial_idxes_by_domain.keys())
            while num_aug_dials > 0:
                random.shuffle(all_domains)
                i = random.choices([2,3,4], weights=[0.7, 0.2, 0.1])[0]
                dials_idx = [random.choice(single_domain_dial_idxes_by_domain[domain]) for domain in all_domains[:i]]
                dials = [single_domain_dials[idx] for idx in dials_idx]
                dial = merge_single_dials(dials)
                dial['dials_idx'] = [int(idx) for idx in dials_idx]
                aug_dials.append(dial)
                num_aug_dials -= len(dial['domains'])

        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCAT2REL:
            slot_pairs = kwargs['slot_pairs']
            slots = sorted(set(['-'.join(slot_pair.split('-')[i*2:(i+1)*2]) for slot_pair in slot_pairs for i in [0,1]]))
            domains = list(single_domain_dial_idxes_by_domain.keys())
            # create a table for query
            dial_idx2state = pd.DataFrame([],index=list(range(len(single_domain_dials))), columns=slots)
            for idx, dial in enumerate(single_domain_dials):
                state = {}
                assert len(dial['domains']) == 1
                domain = dial['domains'][0]
                # get final state
                for i in range(len(dial['turns'])-1,-1,-1):
                    if 'state' in dial['turns'][i]:
                        state = dial['turns'][i]['state']
                        break
                dial['state'] = state
                
                if len(state) == 0:
                    continue
                for slot in state[domain]:
                    domain_slot = f'{domain}-{slot}'
                    # record the non-empty slots for this dialog
                    if domain_slot in slots:
                        dial_idx2state.loc[idx, domain_slot] = True
            dial_idx2state.to_csv('test.csv')
            while num_aug_dials > 0:
                first_domain = random.choice(domains)
                dials_idx = [random.choice(single_domain_dial_idxes_by_domain[first_domain])]
                first_dial = single_domain_dials[dials_idx[0]]
                # only dialogs with possible coref slots could be used as first dialog
                if 'qa' not in first_dial:
                    continue
                # select second dial
                candidate_slots = []
                for second_domain in first_dial['qa']:
                    for dst_slot in first_dial['qa'][second_domain]:
                        candidate_slots.append(f'{second_domain}-{dst_slot}')
                # random sample from any cross-domain dialogs that have non-empty slots in candidates
                mask = dial_idx2state[candidate_slots].notnull().any(axis=1)
                candidate_idxes = dial_idx2state[mask].index
                if len(candidate_idxes) > 0:
                    dials_idx.append(random.choice(candidate_idxes))
                    second_dial = single_domain_dials[dials_idx[1]]
                    second_domain = second_dial['domains'][0]
                    dial = merge_single_domain_rel_dials(first_dial, second_dial)
                    dial['dials_idx'] = dials_idx
                    dial['dials_idx'] = [int(idx) for idx in dial['dials_idx']]
                    # find the turn to be rewritten
                    for turn_idx, turn in enumerate(dial['turns']):
                        if 'replace_state' in turn and 'utterance4rewrite' in turn:
                            # only modify user utt
                            dial['target_turn_idx'] = turn_idx
                            break
                    aug_dials.append(dial)
                    num_aug_dials -= 2
        
        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCATNREL:
            slot_pairs = kwargs['slot_pairs']
            slots = sorted(set(['-'.join(slot_pair.split('-')[i*2:(i+1)*2]) for slot_pair in slot_pairs for i in [0,1]]))
            domains = list(single_domain_dial_idxes_by_domain.keys())
            # create a table for query
            dial_idx2state = pd.DataFrame([],index=list(range(len(single_domain_dials))), columns=slots)
            for idx, dial in enumerate(single_domain_dials):
                state = {}
                assert len(dial['domains']) == 1
                domain = dial['domains'][0]
                # get final state
                for i in range(len(dial['turns'])-1,-1,-1):
                    if 'state' in dial['turns'][i]:
                        state = dial['turns'][i]['state']
                        break
                dial['state'] = state
                
                if len(state) == 0:
                    continue
                for slot in state[domain]:
                    domain_slot = f'{domain}-{slot}'
                    # record the non-empty slots for this dialog
                    if domain_slot in slots:
                        dial_idx2state.loc[idx, domain_slot] = True
            dial_idx2state.to_csv('test.csv')
            while num_aug_dials > 0:
                dom_cnt = random.choices([2,3,4], weights=[0.7,0.2,0.1])[0]
                while True:
                    # first dialog, must have possible target slots
                    domain = random.choice(domains)
                    dial_idx = random.choice(single_domain_dial_idxes_by_domain[domain])
                    dial = single_domain_dials[dial_idx]
                    if 'qa' not in dial:
                        continue

                    active_domains = [domain]
                    active_dials_idx = [dial_idx]
                    active_dials = [dial]
                    while len(active_domains) < dom_cnt:
                        # append a single-domain dialog
                        candidate_slots = []
                        # possible target slots of new domain dial can refer to all prev dials
                        for dial in active_dials:
                            if 'qa' not in dial:
                                continue
                            for dst_domain in dial['qa']:
                                if dst_domain not in active_domains:
                                    for dst_slot in dial['qa'][dst_domain]:
                                        candidate_slots.append(f'{dst_domain}-{dst_slot}')

                        # random sample from any cross-domain dialogs that have non-empty slots in candidates
                        mask = dial_idx2state[candidate_slots].notnull().any(axis=1)
                        candidate_idxes = dial_idx2state[mask].index
                        if len(candidate_idxes) > 0:
                            dial_idx = random.choice(candidate_idxes)
                            dial = single_domain_dials[dial_idx]
                            domain = dial['domains'][0]
                            active_dials_idx.append(dial_idx)
                            active_dials.append(dial)
                            active_domains.append(domain)
                        else:
                            break
                    if len(active_domains) < dom_cnt:
                        # not enough dials for concat
                        continue
                    assert len(active_dials) == dom_cnt
                    dial = mergeN_single_domain_rel_dials(active_dials)
                    dial['dials_idx'] = active_dials_idx
                    dial['dials_idx'] = [int(idx) for idx in dial['dials_idx']]
                    aug_dials.append(dial)
                    num_aug_dials -= dom_cnt
                    break

        elif self.data_aug_type in [DataProcessor.AUG_TYPE_CONCAT2ANA, DataProcessor.AUG_TYPE_CONCATNANA]:
            coqr_slot_pairs = kwargs['coqr_slot_pairs']
            for dial in multi_domain_dials:
                for turn in dial['turns']:
                    if 'coqr_utterance' in turn:
                        turn['utterance'] = turn['coqr_utterance']
                        assert 'replace_state' in turn
                        for slot_pair in turn['replace_state']:
                            src_d, src_s, _ = slot_pair['source']
                            tgt_d, tgt_s, _ = slot_pair['target']
                            k = str((src_d, src_s, tgt_d, tgt_s))
                            coqr_slot_pairs.setdefault(k, 0)
                            coqr_slot_pairs[k] += 1
                aug_dials.append(dial)
                num_aug_dials -= len(dial['domains'])
                if num_aug_dials <= 0:
                    break
            
        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCAT2ANA_UNIFORM:
            aug_dials = []
            slot_pair2dials = {}
            for dial in multi_domain_dials:
                for turn in dial['turns']:
                    if 'coqr_utterance' in turn:
                        turn['utterance'] = turn['coqr_utterance']
                        for slot_pair in turn['replace_state']:
                            src_d, src_s, _ = slot_pair['source']
                            tgt_d, tgt_s, _ = slot_pair['target']
                            k = str((src_d, src_s, tgt_d, tgt_s))
                            slot_pair2dials.setdefault(k, [])
                            slot_pair2dials[k].append(dial)
            coqr_slot_pairs = kwargs['coqr_slot_pairs']
            th = 0
            for idx, (slot_pair, dials) in enumerate(sorted(slot_pair2dials.items(),key=lambda x: len(x[1]))):
                coqr_slot_pairs.setdefault(slot_pair, 0)
                th = (num_aug_dials//2-len(aug_dials))//(len(slot_pair2dials)-idx)
                if th > len(dials):
                    aug_dials.extend(dials)
                    coqr_slot_pairs[slot_pair] += len(dials)
                else:
                    aug_dials.extend(dials[:th])
                    coqr_slot_pairs[slot_pair] += th
                            

        elif self.data_aug_type in [DataProcessor.AUG_TYPE_10TRUE_AND_SYN, DataProcessor.AUG_TYPE_15TRUE_AND_SYN, 
                                    DataProcessor.AUG_TYPE_10TRUE, DataProcessor.AUG_TYPE_15TRUE, DataProcessor.AUG_TYPE_TRUE_AND_SYN]:
            syn_multi_domain_dials = kwargs['syn_multi_domain_dials']
            coqr_slot_pairs = kwargs['coqr_slot_pairs']
                
            if self.data_aug_type in [DataProcessor.AUG_TYPE_10TRUE_AND_SYN, DataProcessor.AUG_TYPE_10TRUE]:
                true_domains = ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1']
            elif self.data_aug_type in [DataProcessor.AUG_TYPE_15TRUE_AND_SYN, DataProcessor.AUG_TYPE_15TRUE]:
                true_domains = ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
                                'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2']
            else:
                true_domains = ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
                                'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2', \
                                'Alarm_1', 'Buses_3', 'Movies_1', 'RentalCars_3']
            
            # replace single domain dials with true multi-domain dials
            random.shuffle(multi_domain_dials)
            for domain in single_domain_dial_idxes_by_domain:
                random.shuffle(single_domain_dial_idxes_by_domain[domain])
            remove_idxes = []
            i = 0
            aug_dials = []
            num_true_dials = round(len(single_domain_dials) * 0.1)
            while num_true_dials > 0:
                if i >= len(multi_domain_dials):
                    raise IndexError("not enough true multi-domain dials for replacement")
                dial = multi_domain_dials[i]
                i += 1
                domains = dial['domains']
                # skip if one of the domain is not accessible for multi-domain dialog
                if any([domain not in true_domains for domain in domains]):
                    continue
                # skip when single domain dialogs are consumed
                if any([len(single_domain_dial_idxes_by_domain[domain]) == 0 for domain in domains]):
                    continue
                # cnt a single domain dialog in each domain
                dial['dials_idx'] = []
                for domain in domains:
                    remove_idxes.append(single_domain_dial_idxes_by_domain[domain].pop())
                    dial['dials_idx'].append(remove_idxes[-1])
                aug_dials.append(dial)
                num_true_dials -= len(domains)
            # remove the single domain dialogs as they are traded for multi-domain dialogs
            for dial_idx in sorted(remove_idxes, reverse=True):
                del single_domain_dials[dial_idx]

            if self.data_aug_type in [DataProcessor.AUG_TYPE_10TRUE_AND_SYN, DataProcessor.AUG_TYPE_15TRUE_AND_SYN, DataProcessor.AUG_TYPE_TRUE_AND_SYN]:
                # add syn multi-domain dialogs where one of the domain is not in true domains
                for dial in syn_multi_domain_dials:
                    if self.data_aug_type != DataProcessor.AUG_TYPE_TRUE_AND_SYN and \
                        all([domain in true_domains for domain in dial['domains']]):
                        # skip domain combinations that have true multi-domain data
                        continue
                    if any([idx in remove_idxes for idx in dial['dials_idx']]):
                        # skip removed single domain dials
                        continue
                    for turn in dial['turns']:
                        if 'coqr_utterance' in turn:
                            turn['utterance'] = turn['coqr_utterance']
                            assert 'replace_state' in turn
                            for slot_pair in turn['replace_state']:
                                src_d, src_s, _ = slot_pair['source']
                                tgt_d, tgt_s, _ = slot_pair['target']
                                k = str((src_d, src_s, tgt_d, tgt_s))
                                coqr_slot_pairs.setdefault(k, 0)
                                coqr_slot_pairs[k] += 1
                    aug_dials.append(dial)
                    num_aug_dials -= len(dial['domains'])
                    if num_aug_dials <= 0:
                        break
        
        elif self.data_aug_type in [DataProcessor.AUG_TYPE_10SYN, DataProcessor.AUG_TYPE_15SYN]:
            if self.data_aug_type == DataProcessor.AUG_TYPE_10SYN:
                syn_domains = ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1']
            elif self.data_aug_type == DataProcessor.AUG_TYPE_15SYN:
                syn_domains = ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
                                'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2']
            coqr_slot_pairs = kwargs['coqr_slot_pairs']
            for dial in multi_domain_dials:
                if not all([domain in syn_domains for domain in dial['domains']]):
                    continue
                for turn in dial['turns']:
                    if 'coqr_utterance' in turn:
                        turn['utterance'] = turn['coqr_utterance']
                        assert 'replace_state' in turn
                        for slot_pair in turn['replace_state']:
                            src_d, src_s, _ = slot_pair['source']
                            tgt_d, tgt_s, _ = slot_pair['target']
                            k = str((src_d, src_s, tgt_d, tgt_s))
                            coqr_slot_pairs.setdefault(k, 0)
                            coqr_slot_pairs[k] += 1
                aug_dials.append(dial)
                num_aug_dials -= len(dial['domains'])
                if num_aug_dials <= 0:
                    break

        
        elif self.data_aug_type in [DataProcessor.AUG_TYPE_CONCAT2ANA, DataProcessor.AUG_TYPE_CONCAT2GPT]:
            aug_dials = multi_domain_dials[:num_aug_dials//2]
            for dial in aug_dials:
                for turn in dial['turns']:
                    if 'coqr_utterance' in turn:
                        turn['utterance'] = turn['coqr_utterance']

        elif self.data_aug_type == DataProcessor.AUG_TYPE_CONCAT2ELL:
            aug_dials = multi_domain_dials[:num_aug_dials//2]
            for dial in aug_dials:
                for turn in dial['turns']:
                    if 'elli_utterance' in turn:
                        turn['utterance'] = turn['elli_utterance']

        elif self.data_aug_type in [DataProcessor.AUG_TYPE_CONCAT2MIX, DataProcessor.AUG_TYPE_CONCATNMIX]:
            coqr_samples = kwargs['coqr_samples']
            elli_samples = kwargs['elli_samples']
            selected_dials_idx = set()
            aug_dials = []
            dial_cnt = 0
            for sample in coqr_samples:
                dial_idx = sample['dial_idx']
                selected_dials_idx.add(dial_idx)
                dial = multi_domain_dials[dial_idx]
                dial['turns'][sample['turn_idx']]['utterance'] = sample['predictions']
                dial['turns'][sample['turn_idx']]['coqr_utterance'] = sample['predictions']
                aug_dials.append(dial)
                dial_cnt += len(dial['domains'])
                if dial_cnt > (num_aug_dials//3):
                    break

            for sample in elli_samples:
                dial_idx = sample['dial_idx']
                if dial_idx in selected_dials_idx:
                    continue
                selected_dials_idx.add(dial_idx)
                dial = multi_domain_dials[dial_idx]
                dial['turns'][sample['turn_idx']]['utterance'] = sample['new_utterance']
                dial['turns'][sample['turn_idx']]['elli_utterance'] = sample['new_utterance']
                aug_dials.append(dial)
                dial_cnt += len(dial['domains'])
                if dial_cnt > (num_aug_dials*2//3):
                    break
            
            for dial_idx, dial in enumerate(multi_domain_dials):
                if dial_idx in selected_dials_idx:
                    continue
                aug_dials.append(dial)
                dial_cnt += len(dial['domains'])
                if dial_cnt > num_aug_dials:
                    break

        return aug_dials


                
    # def data_augmentation_old(self, single_domain_dials, multi_domain_dials, num_aug_times, **kwargs):
    #     random.seed(42)

    #     single_domain_dials_by_domain = {}
    #     for dial in single_domain_dials:
    #         assert len(dial['domains']) == 1
    #         domain = dial['domains'][0]
    #         single_domain_dials_by_domain.setdefault(domain, [])
    #         single_domain_dials_by_domain[domain].append(dial)
        
    #     new_dials = []
    #     num_aug_dials = round(len(single_domain_dials) * num_aug_times)
    #     if self.data_aug_type == 0:
    #         # TRUE multi-domain dials
    #         random.shuffle(multi_domain_dials)
    #         i = 0
    #         while num_aug_dials > 0 and i < len(multi_domain_dials):
    #             dial = multi_domain_dials[i]
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])
    #             i += 1
        
    #     elif self.data_aug_type in [-1, -2]:
    #         random.shuffle(multi_domain_dials)
    #         single_domain_dial_idx = {}
    #         for idx, dial in enumerate(single_domain_dials):
    #             domain = dial['domains'][0]
    #             single_domain_dial_idx.setdefault(domain, [])
    #             single_domain_dial_idx[domain].append(idx)
    #         for domain in single_domain_dial_idx:
    #             random.shuffle(single_domain_dial_idx[domain])
    #         remove_idxes = []
    #         i = 0
    #         while num_aug_dials > 0 and i < len(multi_domain_dials):
    #             dial = multi_domain_dials[i]
    #             i += 1
    #             if any([len(single_domain_dial_idx[domain]) == 0 for domain in dial['domains']]):
    #                 continue
    #             dials = []
    #             for domain in dial['domains']:
    #                 remove_idxes.append(single_domain_dial_idx[domain].pop(-1))

    #                 if self.data_aug_type == -2:
    #                     dials.append(single_domain_dials[remove_idxes[-1]])
                    
    #             if self.data_aug_type == -2:
    #                 dial = merge_single_dials(dials)
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])
                
    #         remove_idxes = sorted(remove_idxes, reverse=True)
    #         for idx in remove_idxes:
    #             single_domain_dials.pop(idx)

    #     elif self.data_aug_type == 1:
    #         # follow domain combinations of TRUE multi-domain dials, concat single domain dials
    #         random.shuffle(multi_domain_dials)
    #         i = 0
    #         while num_aug_dials > 0:
    #             domains = multi_domain_dials[i]['domains']
    #             dials = [random.choice(single_domain_dials_by_domain[domain]) for domain in domains]
    #             dial = merge_single_dials(dials)
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])
    #             i = i + 1
    #             if i == len(multi_domain_dials):
    #                 i = 0

    #     elif self.data_aug_type == 2:
    #         # concat single domain dials from all domains
    #         while num_aug_dials > 0:
    #             dials = [random.choice(single_domain_dials_by_domain[domain]) for domain in single_domain_dials_by_domain]
    #             dial = merge_single_dials(dials)
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])

    #     elif self.data_aug_type == 3:
    #         # concat dials from 2 randomly chosen domains
    #         all_domains = list(single_domain_dials_by_domain.keys())
    #         while num_aug_dials > 0:
    #             d1, d2 = random.sample(all_domains,2)
    #             dial = merge_single_dials([random.choice(single_domain_dials_by_domain[d1]), random.choice(single_domain_dials_by_domain[d2])])
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])

    #     elif self.data_aug_type == 4:
    #         # with probability p=0.8 to append a dial from a new domain
    #         all_domains = list(single_domain_dials_by_domain.keys())
    #         while num_aug_dials > 0:
    #             random.shuffle(all_domains)
    #             i = 1
    #             while random.random() < 0.8:
    #                 i += 1
    #             dials = [random.choice(single_domain_dials_by_domain[domain]) for domain in all_domains[:i]]
    #             dial = merge_single_dials(dials)
    #             new_dials.append(dial)
    #             num_aug_dials -= len(dial['domains'])

    #     elif self.data_aug_type == 5:
    #         # a sinlge domain dial and another different domain dial with at least 1 slot has same value.
    #         slot_pairs = kwargs['slot_pairs']
    #         slots = sorted(set(['-'.join(slot_pair.split('-')[i*2:(i+1)*2]) for slot_pair in slot_pairs for i in [0,1]]))
    #         domains = list(single_domain_dials_by_domain.keys())
    #         domain2slot = {slot.split('-')[0]: slot.split('-')[1] for slot in slots}
    #         dial_idx2state = pd.DataFrame([],index=list(range(len(single_domain_dials))), columns=slots)
    #         for idx, dial in enumerate(single_domain_dials):
    #             state = {}
    #             domain = dial['domains'][0]
    #             for turn in dial['turns']:
    #                 if 'state' in turn:
    #                     state = turn['state']
    #             dial['state'] = state
    #             assert len(dial['domains']) == 1
    #             if len(state) == 0:
    #                 continue
    #             for slot in state[domain]:
    #                 domain_slot = f'{domain}-{slot}'
    #                 if domain_slot in slots:
    #                     dial_idx2state.loc[idx, domain_slot] = state[domain][slot]
    #             if 'qa' not in dial:
    #                 # cannot be used as first dial
    #                 continue
    #             for dst_domain in list(dial['qa'].keys()):
    #                 for dst_slot in list(dial['qa'][dst_domain].keys()):
    #                     (src_slot, src_value), dst_value = dial['qa'][dst_domain][dst_slot]
    #                     if f'{domain}-{src_slot}-{dst_domain}-{dst_slot}' in slot_pairs and \
    #                             src_value != '<no answer>' and 'no answer' not in dst_value and src_value==dst_value:
    #                         pass
    #                     else:
    #                         dial['qa'][dst_domain].pop(dst_slot)
    #                 if len(dial['qa'][dst_domain]) == 0:
    #                     dial['qa'].pop(dst_domain)
                    
            
    #         while num_aug_dials > 0:
    #             first_domain = random.choice(domains)
    #             first_dial = random.choice(single_domain_dials_by_domain[first_domain])
    #             if 'qa' not in first_dial:
    #                 continue
    #             # select second dial
    #             candidate_slots = []
    #             for second_domain in first_dial['qa']:
    #                 for dst_slot, v in first_dial['qa'][second_domain].items():
    #                     src_slot, src_value = v[0]
    #                     dst_value = v[1]
    #                     if src_value != '<no answer>' and 'no answer' not in dst_value and src_value==dst_value:
    #                         candidate_slots.append(f'{second_domain}-{dst_slot}')
    #             # random sample from any cross-domain
    #             mask = dial_idx2state[candidate_slots].notnull().any(axis=1)
    #             candidate_idxes = dial_idx2state[mask].index
    #             if len(candidate_idxes) > 0:
    #                 second_dial = single_domain_dials[random.choice(candidate_idxes)]
    #                 second_domain = second_dial['domains'][0]
    #                 dial = [first_dial, second_dial]
    #                 # print(first_domain, first_dial['state'])
    #                 # print(first_dial['qa'][second_domain])
    #                 # print([slot for slot in candidate_slots if second_domain in slot])
    #                 # print(second_domain, second_dial['state'])
    #                 new_dials.append(dial)
    #                 num_aug_dials -= 2

    #     elif self.data_aug_type == 6:
    #         # coqr dials for training
    #         new_dials = multi_domain_dials[:num_aug_dials//2]
    #         for dial in new_dials:
    #             for turn in dial['turns']:
    #                 if 'coqr_utterance' in turn:
    #                     turn['utterance'] = turn['coqr_utterance']
    #                 turn['utterance'] = turn['utterance'].replace('<sub>', '').replace('</sub>', '').replace('<del>', '').replace('</del>', '')

    #     elif self.data_aug_type == 7:
    #         # coqr dials for training
    #         new_dials = multi_domain_dials[:num_aug_dials//2]
    #         for dial in new_dials:
    #             for turn in dial['turns']:
    #                 turn['utterance'] = turn['utterance'].replace('<sub>', '').replace('</sub>', '').replace('<del>', '').replace('</del>', '')            

    #     else:
    #         raise ValueError(f"Does not support data augmentation of type {self.data_aug_type}")

    #     return new_dials
    
    # def merge_single_domain_old(self, dials2merge, merge_type, **kwargs):
    #     # merge two dials (`train_single_domain_qa.json`) that have at least one slot-pair shares same value
    #     merged_dials = []
    #     for dial1, dial2 in dials2merge:
    #         new_dial = deepcopy(dial1)
    #         new_dial.pop('qa')
    #         new_dial.pop('state')
    #         domain1, domain2 = dial1['domains'][0], dial2['domains'][0]
    #         state1 = dial1['state'][domain1]
    #         state2 = dial2['state'][domain2]
    #         # find slots in domain2 that can have the same value as the slots in domain1
    #         coref_slots = {}
    #         for slot2, item in dial1['qa'][domain2].items():
    #             if slot2 not in state2:
    #                 continue
    #             slot1 = item[0][0]
    #             value1 = state1[slot1]
    #             slot1_desc = self.ontology['domains'][domain1]['slots'][slot1]['description']
    #             slot1_desc = f' ({slot1_desc.lower()}) '
    #             coref_slots[slot2] = (slot1, slot1_desc, value1)
    #         if len(coref_slots) == 0:
    #             continue
    #         new_dial['domains'] = [domain1, domain2]

    #         # append dial2 turns
    #         prev_state = {}
    #         for turn in dial2['turns']:
    #             new_turn = deepcopy(turn)
    #             if 'state' in turn:
    #                 cur_state = new_turn.pop('state')
    #                 state_update = get_state_update(prev_state, cur_state)
    #                 new_turn['state_update'] = state_update
    #                 prev_state = cur_state
    #             new_dial['turns'].append(new_turn)

    #         # modified the merge dialog by replacing tgt_slot_value with src_slot_value
    #         for tgt_slot, (src_slot, src_slot_desc, src_slot_value) in coref_slots.items():
    #             tgt_slot_appear_turn_idxes = []
    #             for turn_idx, turn in enumerate(new_dial['turns']):
    #                 # find the turn that target slot first appears
    #                 if 'state_update' in turn and domain2 in turn['state_update'] and tgt_slot in turn['state_update'][domain2]:
    #                     tgt_slot_appear_turn_idxes.append(turn_idx)
    #                     if len(tgt_slot_appear_turn_idxes) == 1:
    #                         # first time
    #                         tgt_slot_value = turn['state_update'][domain2][tgt_slot]
    #                         turn['replaced_state'] = {'target':[domain2, tgt_slot, tgt_slot_value], 
    #                             'source': [domain1, src_slot, src_slot_value]}
    #                         turn['state_update'][domain2][tgt_slot] = src_slot_value
    #                     else:
    #                         # second time
    #                         break
    #             start_turn= max(0, tgt_slot_appear_turn_idxes[0]-1)
    #             if len(tgt_slot_appear_turn_idxes) > 1:
    #                 end_turn = tgt_slot_appear_turn_idxes[1]
    #             else:
    #                 end_turn = len(new_dial['turns']) - 1
    #             for turn_idx in range(start_turn, end_turn+1):
    #                 utt = new_dial['turns'][turn_idx]['utterance']
    #                 coref_utt = find_and_replace_value_in_sen(tgt_slot_value, utt, f'<sub>{src_slot_value}</sub>')
    #                 if coref_utt is not None:
    #                     new_dial['turns'][turn_idx]['utterance'] = coref_utt
    #                     new_dial['turns'][turn_idx]['replaced_utt'] = utt

    #         for turn_idx, turn in enumerate(new_dial['turns']):
    #             if 'state_update' in turn:
    #                 new_state = update_state(new_dial['turns'][turn_idx-2]['state'], turn.pop('state_update'))
    #                 turn['state'] = new_state

    #         merged_dials.append(new_dial)
    #     return merged_dials
