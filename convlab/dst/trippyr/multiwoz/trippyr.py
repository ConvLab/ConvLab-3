# Copyright 2021 Heinrich Heine University Duesseldorf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import copy
import pickle

import torch
from transformers import (RobertaConfig, RobertaTokenizer)

from convlab2.dst.trippyr.multiwoz.modeling_roberta_dst import (RobertaForDST)

from convlab2.dst.dst import DST
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import normalize_value

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForDST, RobertaTokenizer),
}


TEMPLATE_STATE = {
    "attraction": {
        "type": "",
        "name": "",
        "area": ""
    },
    "hotel": {
        "name": "",
        "area": "",
        "parking": "",
        "price range": "",
        "stars": "",
        "internet": "",
        "type": "",
        "book stay": "",
        "book day": "",
        "book people": ""
    },
    "restaurant": {
        "food": "",
        "price range": "",
        "name": "",
        "area": "",
        "book time": "",
        "book day": "",
        "book people": ""
    },
    "taxi": {
        "leave at": "",
        "destination": "",
        "departure": "",
        "arrive by": ""
    },
    "train": {
        "leave at": "",
        "destination": "",
        "day": "",
        "arrive by": "",
        "departure": "",
        "book people": ""
    },
    "hospital": {
        "department": ""
    }
}


SLOT_MAP_TRIPPY_TO_UDF = {
    'hotel': {
        'pricerange': 'price range',
        'book_stay': 'book stay',
        'book_day': 'book day',
        'book_people': 'book people',
        'addr': 'address',
        'post': 'postcode'
    },
    'restaurant': {
        'pricerange': 'price range',
        'book_time': 'book time',
        'book_day': 'book day',
        'book_people': 'book people',
        'addr': 'address',
        'post': 'postcode'
    },
    'taxi': {
        'arriveBy': 'arrive by',
        'leaveAt': 'leave at',
        'arrive': 'arrive by',
        'leave': 'leave at',
        'car': 'type',
        'car type': 'type',
        'depart': 'departure',
        'dest': 'destination'
    },
    'train': {
        'arriveBy': 'arrive by',
        'leaveAt': 'leave at',
        'book_people': 'book people',
        'arrive': 'arrive by',
        'leave': 'leave at',
        'depart': 'departure',
        'dest': 'destination',
        'id': 'train id',
        'people': 'book people',
        'time': 'duration',
        'ticket': 'price',
        'trainid': 'train id'
    },
    'attraction': {
        'post': 'postcode',
        'addr': 'address',
        'fee': 'entrance fee',
        'price': 'price range'
    },
    'general': {},
    'hospital': {
        'post': 'postcode',
        'addr': 'address'
    },
    'police': {
        'post': 'postcode',
        'addr': 'address'
    }
}


class TRIPPYR(DST):
    def print_header(self):
        print(" _________  ________  ___  ________  ________  ___    ___        ________     ")
        print("|\___   ___\\\   __  \|\  \|\   __  \|\   __  \|\  \  /  /|      |\   __  \    ")
        print("\|___ \  \_\ \  \|\  \ \  \ \  \|\  \ \  \|\  \ \  \/  / /______\ \  \|\  \   ")
        print("     \ \  \ \ \   _  _\ \  \ \   ____\ \   ____\ \    / /\_______\ \   _  _\  ")
        print("      \ \  \ \ \  \\\  \\\ \  \ \  \___|\ \  \___|\/  /  /\|_______|\ \  \\\  \| ")
        print("       \ \__\ \ \__\\\ _\\\ \__\ \__\    \ \__\ __/  / /             \ \__\\\ _\ ")
        print("        \|__|  \|__|\|__|\|__|\|__|     \|__||\___/ /               \|__|\|__|")
        print("          (c) 2022 Heinrich Heine University \|___|/                          ")
        print()

    def __init__(self, model_type="roberta", model_name="roberta-base", model_path="", nlu_path="", emb_path="", fp16=False):
        super(TRIPPYR, self).__init__()

        self.print_header()

        self.model_type = model_type.lower()
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.nlu_path = nlu_path

        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_type]
        self.config = self.config_class.from_pretrained(self.model_path)
        # TODO: update config (parameters)

        self.load_weights()
        self.load_embeddings(emb_path, fp16)
    
    def load_weights(self):
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name) # TODO: do_lower_case=args.do_lower_case ?
        self.model = self.model_class.from_pretrained(self.model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        self.nlu = BERTNLU(model_file=self.nlu_path) # TODO: remove, once TripPy takes over its task

    def load_embeddings(self, emb_path, fp16=False):
        self.encoded_slots_pooled = pickle.load(open(os.path.join(emb_path, "encoded_slots_pooled.pickle"), "rb"))
        self.encoded_slots_seq = pickle.load(open(os.path.join(emb_path, "encoded_slots_seq.pickle"), "rb"))
        self.encoded_slot_values = pickle.load(open(os.path.join(emb_path, "encoded_slot_values_test.pickle"), "rb"))
        if fp16:
            for e in self.encoded_slots_pooled:
                self.encoded_slots_pooled[e] = self.encoded_slots_pooled[e].type(torch.float32)
            for e in self.encoded_slots_seq:
                self.encoded_slots_seq[e] = self.encoded_slots_seq[e].type(torch.float32)
            for e in self.encoded_slot_values:
                for f in self.encoded_slot_values[e]:
                    self.encoded_slot_values[e][f] = self.encoded_slot_values[e][f].type(torch.float32)

    def init_session(self):
        self.state = default_state() # Initialise as empty state
        self.state['belief_state'] = copy.deepcopy(TEMPLATE_STATE)
        # TODO: define internal variables here as well that are tracked but not forwarded

    def update(self, user_act=''):
        def filter_sequences(seqs, mode="first"):
            if mode == "first":
                return tokenize(seqs[0][0][0])
            elif mode == "max_first":
                max_conf = 0
                max_idx = 0
                for e_itr, e in enumerate(seqs[0]):
                    if e[1] > max_conf:
                        max_conf = e[1]
                        max_idx = e_itr
                return tokenize(seqs[0][max_idx][0])
            elif mode == "max":
                max_conf = 0
                max_t_idx = 0
                for t_itr, t in enumerate(seqs):
                    for e_itr, e in enumerate(t):
                        if e[1] > max_conf:
                            max_conf = e[1]
                            max_t_idx = t_itr
                            max_idx = e_itr
                return tokenize(seqs[max_t_idx][max_idx][0])
            else:
                print("WARN: mode %s unknown. Aborting." % mode)
                exit()

        def tokenize(text):
            if "\u0120" in text:
                text = re.sub(" ", "", text)
                text = re.sub("\u0120", " ", text)
                text = text.strip()
            return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])

        prev_state = self.state

        # TODO: add asserts to check format. if wrong, print suggested config
        #print("--")

        # --- Get inform memory and auxiliary features ---

        # If system_action is plain text, get acts using NLU
        if isinstance(prev_state['system_action'], str):
            acts, _ = self.get_acts(prev_state['system_action'])
        elif isinstance(prev_state['system_action'], list):
            acts = prev_state['system_action']
        else:
            raise Exception('Unknown format for system action:', prev_state['system_action'])
        inform_mem = self.get_inform_mem(acts)
        printed_inform_mem = False
        for s in inform_mem:
            if inform_mem[s] != 'none':
                if not printed_inform_mem:
                    print("DST: inform_mem:")
                print(s, ':', inform_mem[s])
                printed_inform_mem = True

        # --- Tokenize dialogue context and feed DST model ---

        features = self.get_features(self.state['history'])
        pred_states, pred_classes, cls_representation = self.predict(features, inform_mem)
        #print(pred_states)
        #print(pred_classes)
        #import pdb
        #pdb.set_trace()

        # --- Update ConvLab-style dialogue state ---

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        user_acts = []
        for state, value in pred_states.items():
            if isinstance(value, list):
                value = filter_sequences(value, mode="max")
            else:
                value = tokenize(value)
            #if pred_classes[state] > 0:
            #    print(pred_classes[state], state, value)

            domain, slot = state.split('-', 1)

            # 1.) Domain prediction
            if slot == "none": # and pred_classes[state] == 3:
                continue # for now, continue

            # 2.) Requests and greetings
            if pred_classes[state] == 7:
                if domain == "general":
                    user_acts.append([slot, 'general', 'none', 'none'])
                else:
                    user_acts.append(['request', domain, SLOT_MAP_TRIPPY_TO_UDF[domain].get(slot, slot), '?'])

            # 3.) Informable slots
            if value == 'none':
                continue
            # Value normalization # TODO: according to trippy rules?
            if domain == 'hotel' and slot == 'type':
                value = "hotel" if value == "yes" else "guesthouse"
            value = normalize_value(self.value_dict, domain, slot, value)
            # TODO: needed?
            if domain not in new_belief_state:
                if domain == 'bus':
                    continue
                else:
                    raise Exception('Domain <{}> not in belief state'.format(domain))
            slot = SLOT_MAP_TRIPPY_TO_UDF[domain].get(slot, slot)
            if slot in new_belief_state[domain]:
                new_belief_state[domain][slot] = value # TODO: value normalization?
                user_acts.append(['inform', domain, SLOT_MAP_TRIPPY_TO_UDF[domain].get(slot, slot), value]) # TODO: value normalization?
            else:
                raise Exception('Unknown slot name <{}> with value <{}> of domain <{}>'.format(slot, value, domain))

        # Update request_state
        #new_request_state = copy.deepcopy(prev_state['request_state'])

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        #new_state['request_state'] = new_request_state

        nlu_user_acts, nlu_system_acts = self.get_acts(user_act)
        for e in nlu_user_acts:
            nlu_a, nlu_d, nlu_s, nlu_v = e
            nlu_a = nlu_a.lower()
            nlu_d = nlu_d.lower()
            nlu_s = nlu_s.lower()
            nlu_v = nlu_v.lower()
            if nlu_a != 'inform':
                user_acts.append([nlu_a, nlu_d, SLOT_MAP_TRIPPY_TO_UDF[nlu_d].get(nlu_s, nlu_s), nlu_v])
        #new_state['system_action'] = nlu_system_acts # Empty when DST for user -> needed?
        new_state['user_action'] = user_acts

        #new_state['cls_representation'] = cls_representation # TODO: needed by Nunu?

        self.state = new_state

        #print("--")
        return self.state
    
    def predict(self, features, inform_mem):
        def _tokenize(text):
            if "\u0120" in text:
                text = re.sub(" ", "", text)
                text = re.sub("\u0120", " ", text)
                text = text.strip()
            return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])

        def get_spans(pred, norm_logits, input_tokens, usr_utt_spans):
            span_indices = [i for i in range(len(pred)) if pred[i]]
            prev_si = None
            spans = []
            #confs = []
            for si in span_indices:
                if prev_si is None or si - prev_si > 1:
                    spans.append(([], [], []))
                    #confs.append([])
                #spans[-1].append(input_tokens[si])
                spans[-1][0].append(si)
                spans[-1][1].append(input_tokens[si])
                spans[-1][2].append(norm_logits[si])
                #confs[-1].append(norm_logits[si])
                prev_si = si
            #spans = [' '.join(t for t in s) for s in spans]
            spans = [(min(i), max(i), ' '.join(t for t in s), (sum(c) / len(c)).item()) for (i, s, c) in spans]
            #confs = [(sum(c) / len(c)).item() for c in confs]
            final_spans = {}
            for s in spans:
                for us_itr, us in enumerate(usr_utt_spans):
                    if s[0] >= us[0] and s[1] <= us[1]:
                        if us_itr not in final_spans:
                            final_spans[us_itr] = []
                        final_spans[us_itr].append(s[2:])
                        break
            final_spans = list(final_spans.values())
            return final_spans # , confs
        
        def get_usr_utt_spans(usr_mask):
            span_indices = [i for i in range(len(usr_mask)) if usr_mask[i]]
            prev_si = None
            spans = []
            for si in span_indices:
                if prev_si is None or si - prev_si > 1:
                    spans.append([])
                spans[-1].append(si)
                prev_si = si
            spans = [[min(s), max(s)] for s in spans]
            return spans

        def smooth_roberta_predictions(pred, input_tokens):
            smoothed_pred = pred.detach().clone()
            # Forward
            span = False
            i = 0
            while i < len(pred):
                if pred[i] > 0:
                    span = True
                elif span and input_tokens[i][0] != "\u0120" and input_tokens[i][0] != "<":
                    smoothed_pred[i] = 1 # TODO: make sure to use label for in-span tokens
                elif span and (input_tokens[i][0] == "\u0120" or input_tokens[i][0] == "<"):
                    span = False
                i += 1
            # Backward
            span = False
            i = len(pred) - 1
            while i >= 0:
                if pred[i] > 0:
                    span = True
                if span and input_tokens[i][0] != "\u0120" and input_tokens[i][0] != "<":
                    smoothed_pred[i] = 1 # TODO: make sure to use label for in-span tokens
                elif span and input_tokens[i][0] == "\u0120":
                    smoothed_pred[i] = 1 # TODO: make sure to use label for in-span tokens
                    span = False
                i -= 1
            #if pred != smoothed_pred:
            #    print(get_spans(pred, input_tokens))
            #    print(get_spans(smoothed_pred, input_tokens))
            return smoothed_pred

        with torch.no_grad():
            outputs = self.model(features) # TODO: mode etc

        input_tokens = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0]) # unmasked!

        # assign identified spans to their respective usr turns (simply append spans as list of lists)
        usr_utt_spans = get_usr_utt_spans(features['usr_mask'][0][1:])

        per_slot_class_logits = outputs[8] # [2]
        per_slot_start_logits = outputs[9] # [3]
        per_slot_end_logits = outputs[10] # [4]
        per_slot_value_logits = outputs[11] # [4]
        per_slot_refer_logits = outputs[12] # [5]

        cls_representation = outputs[16]

        # TODO: maybe add assert to check that batch=1
        
        predictions = {slot: 'none' for slot in self.config.dst_slot_list}
        class_predictions = {slot: 'none' for slot in self.config.dst_slot_list}

        for slot in self.config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][0]
            start_logits = per_slot_start_logits[slot][0]
            end_logits = per_slot_end_logits[slot][0] if slot in per_slot_end_logits else None
            value_logits = per_slot_value_logits[slot][0] if per_slot_value_logits[slot] is not None else None
            refer_logits = per_slot_refer_logits[slot][0]

            weights = start_logits[:len(features['input_ids'][0])]
            norm_logits = torch.clamp(weights - torch.mean(weights), min=0) / max(weights)

            class_prediction = int(class_logits.argmax())
            start_prediction = norm_logits > 0.0
            start_prediction = smooth_roberta_predictions(start_prediction, input_tokens)
            start_prediction[0] = False # Ignore <s>
            end_prediction = int(end_logits.argmax()) if end_logits is not None else None
            refer_prediction = int(refer_logits.argmax())

            if class_prediction == self.config.dst_class_types.index('dontcare'):
                predictions[slot] = 'dontcare'
            elif class_prediction == self.config.dst_class_types.index('copy_value'):
                spans = get_spans(start_prediction[1:], norm_logits[1:], input_tokens[1:], usr_utt_spans)
                if len(spans) > 0:
                    for e_itr in range(len(spans)):
                        for ee_itr in range(len(spans[e_itr])):
                            tmp = list(spans[e_itr][ee_itr])
                            tmp[0] = _tokenize(tmp[0])
                            spans[e_itr][ee_itr] = tuple(tmp)
                    predictions[slot] = spans
                else:
                    predictions[slot] = "none"
            elif 'true' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('true'):
                predictions[slot] = "yes" # 'true'
            elif 'false' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('false'):
                predictions[slot] = "no" # 'false'
            elif class_prediction == self.config.dst_class_types.index('inform'):
                #print("INFORM:", slot, ",", predictions[slot], "->", inform_mem[slot])
                predictions[slot] = inform_mem[slot]
            #elif class_prediction == self.config.dst_class_types.index('request'):
            #    if slot in ["hotel-internet", "hotel-parking"]:
            #        predictions[slot] = "yes" # 'true'
            # Referral case is handled below

            class_predictions[slot] = class_prediction

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in self.config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                predictions[slot] = predictions[list(self.config.dst_slot_list.keys())[refer_prediction]]

        # TODO: value normalization
        # TODO: value matching

            if class_prediction > 0:
                print("  ", slot, "->", class_prediction, ",", predictions[slot])

        return predictions, class_predictions, cls_representation

    def get_features(self, context):
        def to_device(batch, device):
            if isinstance(batch, tuple):
                batch_on_device = tuple([to_device(element, device) for element in batch])
            if isinstance(batch, dict):
                batch_on_device = {k: to_device(v, device) for k, v in batch.items()}
            else:
                batch_on_device = batch.to(device) if batch is not None else batch
            return batch_on_device

        assert(self.model_type == "roberta") # TODO: generalize to other BERT-like models
        input_tokens = [] # ['<s>']
        for e_itr, e in enumerate(reversed(context)):
            input_tokens.append(e[1] if e[1] != 'null' else ' ')
            if e_itr < 2:
                input_tokens.append('</s> </s>')
            else:
                input_tokens.append('</s>')
            # Ignore history for now
            if e_itr == 1:
                break
        if e_itr == 0:
            input_tokens.append('</s> </s>')
        #input_tokens.append('</s>')
        input_tokens = ' '.join(input_tokens)

        # TODO: delex sys utt somehow, or refrain from using delex for sys utts?
        features = self.tokenizer.encode_plus(input_tokens, add_special_tokens=True, max_length=self.config.dst_max_seq_length)

        input_ids = torch.tensor(features['input_ids']).reshape(1,-1)
        input_mask = torch.tensor(features['attention_mask']).reshape(1,-1)
        usr_mask = torch.zeros(input_ids.size())
        usr_seen = False
        sys_seen = False
        usr_sep = 0
        sys_sep = 0
        hst_cnt = 0
        for i_itr, i in enumerate(input_ids[0,:]):
            if i_itr == 0:
                continue
            is_usr = True
            if i == 1:
                is_usr = False
            if i == 2:
                is_usr = False
                if not usr_seen:
                    usr_sep += 1
                    if usr_sep == 2:
                        usr_seen = True
                elif not sys_seen:
                    sys_sep += 1
                    if sys_sep == 2:
                        sys_seen = True
                else:
                    hst_cnt += 1
            if usr_seen and not sys_seen:
                is_usr = False
            elif usr_seen and sys_seen and hst_cnt % 2 == 1:
                is_usr = False
            if is_usr:
                usr_mask[0,i_itr] = 1
        #usr_mask = torch.tensor(features['attention_mask']).reshape(1,-1) # TODO
        features = {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'usr_mask': usr_mask,
                    'start_pos': None,
                    'end_pos': None,
                    'refer_id': None,
                    'class_label_id': None,
                    'inform_slot_id': None,
                    'diag_state': None,
                    'pos_sampling_input': None,
                    'neg_sampling_input': None,
                    'encoded_slots_pooled': self.encoded_slots_pooled,
                    'encoded_slots_seq': self.encoded_slots_seq,
                    'encoded_slot_values': self.encoded_slot_values}

        return to_device(features, self.device)

    # TODO: consider "booked" values?
    def get_inform_mem(self, state):
        inform_mem = {slot: 'none' for slot in self.config.dst_slot_list}
        for e in state:
            a, d, s, v = e
            if a in ['inform', 'recommend', 'select', 'book', 'offerbook']:
                #ds_d = d.lower()
                #if s in REF_SYS_DA[d]:
                #    ds_s = REF_SYS_DA[d][s]
                #elif s in REF_SYS_DA['Booking']:
                #    ds_s = "book_" + REF_SYS_DA['Booking'][s]
                #else:
                #    ds_s = s.lower()
                #    #raise Exception('Slot <{}> of domain <{}> unknown'.format(s, d))
                slot = "%s-%s" % (d, s)
                if slot in inform_mem:
                    inform_mem[slot] = v
        return inform_mem

    # TODO: fix, still a mess...
    def get_acts(self, user_act):
        context = self.state['history']
        if context:
            if context[-1][0] != 'sys':
                system_act = ''
                context = [t for s,t in context]
            else:
                system_act = context[-1][-1]
                context = [t for s,t in context[:-1]]
        else:
            system_act = ''
            context = ['']

        #print("  SYS:", system_act, context)
        system_acts = self.nlu.predict(system_act, context=context)

        context.append(system_act)
        #print("  USR:", user_act, context)
        user_acts = self.nlu.predict(user_act, context=context)
        
        return user_acts, system_acts


# if __name__ == "__main__":
#     tracker = TRIPPY(model_type='roberta', model_path='/path/to/model',
#                         nlu_path='/path/to/nlu')
#     tracker.init_session()
#     state = tracker.update('hey. I need a cheap restaurant.')
#     tracker.state['history'].append(['usr', 'hey. I need a cheap restaurant.'])
#     tracker.state['history'].append(['sys', 'There are many cheap places, which food do you like?'])
#     state = tracker.update('If you have something Asian that would be great.')
#     tracker.state['history'].append(['usr', 'If you have something Asian that would be great.'])
#     tracker.state['history'].append(['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
#     state = tracker.update('Great. Where are they located?')
#     tracker.state['history'].append(['usr', 'Great. Where are they located?'])
#     print(tracker.state)
