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

import re
import copy

import torch
from transformers import (BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer)

from convlab2.dst.trippy.multiwoz.modeling_bert_dst import (BertForDST)
from convlab2.dst.trippy.multiwoz.modeling_roberta_dst import (RobertaForDST)

from convlab2.dst.dst import DST
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab2.nlu.jointBERT.multiwoz import BERTNLU

import pdb


MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
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


class TRIPPY(DST):
    def print_header(self):
        print(" _________  ________  ___  ________  ________  ___    ___ ")
        print("|\___   ___\\\   __  \|\  \|\   __  \|\   __  \|\  \  /  /|")
        print("\|___ \  \_\ \  \|\  \ \  \ \  \|\  \ \  \|\  \ \  \/  / /")
        print("     \ \  \ \ \   _  _\ \  \ \   ____\ \   ____\ \    / / ")
        print("      \ \  \ \ \  \\\  \\\ \  \ \  \___|\ \  \___|\/  /  /  ")
        print("       \ \__\ \ \__\\\ _\\\ \__\ \__\    \ \__\ __/  / /    ")
        print("        \|__|  \|__|\|__|\|__|\|__|     \|__||\___/ /     ")
        print("          (c) 2022 Heinrich Heine University \|___|/      ")
        print()
        
    def __init__(self, model_type="roberta", model_name="roberta-base", model_path="", nlu_path=""):
        super(TRIPPY, self).__init__()

        self.print_header()

        self.model_type = model_type.lower()
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.nlu_path = nlu_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_type]
        self.config = self.config_class.from_pretrained(self.model_path)
        # TODO: update config (parameters)

        self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}

        self.load_weights()
    
    def load_weights(self):
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name) # TODO: do_lower_case=args.do_lower_case ?
        self.model = self.model_class.from_pretrained(self.model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        self.nlu = BERTNLU(model_file=self.nlu_path) # TODO: remove, once TripPy takes over its task

    def init_session(self):
        self.state = default_state() # Initialise as empty state
        self.state['belief_state'] = copy.deepcopy(TEMPLATE_STATE)
        # TODO: define internal variables here as well that are tracked but not forwarded

    def update(self, user_act=''):
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
        inform_aux, inform_mem = self.get_inform_aux(acts)
        printed_inform_mem = False
        for s in inform_mem:
            if inform_mem[s] != 'none':
                if not printed_inform_mem:
                    print("DST: inform_mem:")
                print(s, ':', inform_mem[s])
                printed_inform_mem = True

        # --- Tokenize dialogue context and feed DST model ---

        features = self.get_features(self.state['history'], ds_aux=self.ds_aux, inform_aux=inform_aux)
        pred_states, cls_representation = self.predict(features, inform_mem)

        # --- Update ConvLab-style dialogue state ---

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        user_acts = []
        for state, value in pred_states.items():
            if value == 'none':
                continue
            domain, slot = state.split('-', 1)
            # TODO: value normalizations?
            if domain == 'hotel' and slot == 'type':
                value = "hotel" if value == "yes" else "guesthouse"
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

        # TODO: only for debugging for now!
        if re.search("not.*book.*", prev_state['user_action']) is not None:
            user_acts.append(['inform', 'train', 'notbook', 'none'])

        # Update request_state
        #new_request_state = copy.deepcopy(prev_state['request_state'])

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        #new_state['request_state'] = new_request_state

        # Get requestable slots from NLU, until implemented in DST.
        nlu_user_acts, nlu_system_acts = self.get_acts(user_act)
        for e in nlu_user_acts:
            nlu_a, nlu_d, nlu_s, nlu_v = e
            nlu_a = nlu_a.lower()
            nlu_d = nlu_d.lower()
            nlu_s = nlu_s.lower()
            nlu_v = nlu_v.lower()
            if nlu_a != 'inform':
                user_acts.append([nlu_a, nlu_d, SLOT_MAP_TRIPPY_TO_UDF[nlu_d].get(nlu_s, nlu_s), nlu_v])
        new_state['system_action'] = nlu_system_acts # Empty when DST for user -> needed?
        new_state['user_action'] = user_acts

        #new_state['cls_representation'] = cls_representation # TODO: needed by Nunu?

        self.state = new_state

        # (Re)set internal states
        if self.state['terminated']:
            #print("!!! RESET DS_AUX")
            self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        else:
            self.ds_aux = self.update_ds_aux(self.state['belief_state'])
        #print("ds:", [self.ds_aux[s][0].item() for s in self.ds_aux])

        #print("--")
        return self.state
    
    def predict(self, features, inform_mem):
        with torch.no_grad():
            outputs = self.model(input_ids=features['input_ids'],
                                 input_mask=features['attention_mask'],
                                 inform_slot_id=features['inform_slot_id'],
                                 diag_state=features['diag_state'])

        input_tokens = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0]) # unmasked!

        #total_loss = outputs[0]
        #per_slot_per_example_loss = outputs[1]
        per_slot_class_logits = outputs[2]
        per_slot_start_logits = outputs[3]
        per_slot_end_logits = outputs[4]
        per_slot_refer_logits = outputs[5]

        cls_representation = outputs[6]
            
        # TODO: maybe add assert to check that batch=1
        
        predictions = {slot: 'none' for slot in self.config.dst_slot_list}

        for slot in self.config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][0]
            start_logits = per_slot_start_logits[slot][0]
            end_logits = per_slot_end_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if class_prediction == self.config.dst_class_types.index('dontcare'):
                predictions[slot] = 'dontcare'
            elif class_prediction == self.config.dst_class_types.index('copy_value'):
                predictions[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                predictions[slot] = re.sub("(^| )##", "", predictions[slot])
                if "\u0120" in predictions[slot]:
                    predictions[slot] = re.sub(" ", "", predictions[slot])
                    predictions[slot] = re.sub("\u0120", " ", predictions[slot])
                    predictions[slot] = predictions[slot].strip()
            elif 'true' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('true'):
                predictions[slot] = "yes" # 'true'
            elif 'false' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('false'):
                predictions[slot] = "no" # 'false'
            elif class_prediction == self.config.dst_class_types.index('inform'):
                #print("INFORM:", slot, ",", predictions[slot], "->", inform_mem[slot])
                predictions[slot] = inform_mem[slot]
            # Referral case is handled below

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
                predictions[slot] = predictions[self.config.dst_slot_list[refer_prediction - 1]]

            if class_prediction > 0:
                print("  ", slot, "->", class_prediction, ",", predictions[slot])

        return predictions, cls_representation

    def get_features(self, context, ds_aux=None, inform_aux=None):
        assert(self.model_type == "roberta") # TODO: generalize to other BERT-like models
        input_tokens = ['<s>']
        for e_itr, e in enumerate(reversed(context)):
            #input_tokens.append(e[1].lower() if e[1] != 'null' else ' ') # TODO: normalise text
            input_tokens.append(e[1] if e[1] != 'null' else ' ') # TODO: normalise text
            if e_itr < 2:
                input_tokens.append('</s> </s>')
        if e_itr == 0:
            input_tokens.append('</s> </s>')
        input_tokens.append('</s>')
        input_tokens = ' '.join(input_tokens)

        # TODO: delex sys utt somehow, or refrain from using delex for sys utts?
        features = self.tokenizer.encode_plus(input_tokens, add_special_tokens=False, max_length=self.config.dst_max_seq_length)

        input_ids = torch.tensor(features['input_ids']).reshape(1,-1).to(self.device)
        attention_mask = torch.tensor(features['attention_mask']).reshape(1,-1).to(self.device)
        features = {'input_ids': input_ids,
                    'attention_mask': attention_mask},
                    'inform_slot_id': inform_aux,
                    'diag_state': ds_aux}

        return features

    def update_ds_aux(self, state, terminated=False):
        ds_aux = copy.deepcopy(self.ds_aux) # TODO: deepcopy necessary? just update class variable?
        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            ds_aux[slot][0] = int(state[d][SLOT_MAP_TRIPPY_TO_UDF[d].get(s, s)] != '')
        return ds_aux

    # TODO: consider "booked" values?
    def get_inform_aux(self, state):
        inform_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        inform_mem = {slot: 'none' for slot in self.config.dst_slot_list}
        for e in state:
            #print(e)
            #pdb.set_trace()
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
                if slot in inform_aux:
                    inform_aux[slot][0] = 1
                    inform_mem[slot] = v
        return inform_aux, inform_mem

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
