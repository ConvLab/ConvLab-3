# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
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
import logging

import torch
from transformers import (BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          ElectraConfig, ElectraTokenizer)

from convlab.dst.dst import DST
from convlab.dst.trippy.modeling_dst import (TransformerForDST)
from convlab.dst.trippy.dataset_interfacer import (create_dataset_interfacer)
from convlab.util import relative_import_module_from_unified_datasets

MODEL_CLASSES = {
    'bert': (BertConfig, TransformerForDST('bert'), BertTokenizer),
    'roberta': (RobertaConfig, TransformerForDST('roberta'), RobertaTokenizer),
    'electra': (ElectraConfig, TransformerForDST('electra'), ElectraTokenizer),
}


class TRIPPY(DST):
    def print_header(self):
        logging.info(" _________  ________  ___  ________  ________  ___    ___ ")
        logging.info("|\___   ___\\\   __  \|\  \|\   __  \|\   __  \|\  \  /  /|")
        logging.info("\|___ \  \_\ \  \|\  \ \  \ \  \|\  \ \  \|\  \ \  \/  / /")
        logging.info("     \ \  \ \ \   _  _\ \  \ \   ____\ \   ____\ \    / / ")
        logging.info("      \ \  \ \ \  \\\  \\\ \  \ \  \___|\ \  \___|\/  /  /  ")
        logging.info("       \ \__\ \ \__\\\ _\\\ \__\ \__\    \ \__\ __/  / /    ")
        logging.info("        \|__|  \|__|\|__|\|__|\|__|     \|__||\___/ /     ")
        logging.info("          (c) 2022 Heinrich Heine University \|___|/      ")
        logging.info("")

    def print_dialog(self, hst):
        logging.info("Dialogue %s, turn %s:" % (self.global_diag_cnt, self.global_turn_cnt))
        for utt in hst[:-2]:
            logging.info("  \033[92m%s\033[0m" % (utt))
        if len(hst) > 1:
            logging.info(" %s" % (hst[-2]))
            logging.info(" %s" % (hst[-1]))

    def print_inform_memory(self, inform_mem):
        logging.info("Inform memory:")
        is_all_none = True
        for s in inform_mem:
            if inform_mem[s] != 'none':
                logging.info("  %s = %s" % (s, inform_mem[s]))
                is_all_none = False
        if is_all_none:
            logging.info("  -")

    def eval_user_acts(self, user_act, user_acts):
        logging.info("User acts:")
        for ua in user_acts:
            if ua not in user_act:
                logging.info("  \033[33m%s\033[0m" % (ua))
            else:
                logging.info("  \033[92m%s\033[0m" % (ua))
        for ua in user_act:
            if ua not in user_acts:
                logging.info("  \033[91m%s\033[0m" % (ua))

    def eval_dialog_state(self, state_updates, new_belief_state):
        logging.info("Dialogue state:")
        for d in self.gt_belief_state:
            logging.info("  %s:" % (d))
            for s in new_belief_state[d]:
                is_printed = False
                is_updated = False
                if state_updates[d][s] > 0:
                    is_updated = True
                log_str = ""
                if is_updated:
                    log_str += "\033[3m"
                if new_belief_state[d][s] != self.gt_belief_state[d][s]:
                    self.global_eval_stats[d][s]['FP'] += 1
                    if self.gt_belief_state[d][s] == '':
                        log_str += "    \033[33m%s: %s\033[0m" % (s, new_belief_state[d][s])
                    else:
                        log_str += "    \033[91m%s: %s\033[0m (label: %s)" % (s, new_belief_state[d][s] if new_belief_state[d][s] != '' else 'none', self.gt_belief_state[d][s])
                        self.global_eval_stats[d][s]['FN'] += 1
                    is_printed = True
                elif new_belief_state[d][s] != '':
                    log_str += "    \033[92m%s: %s\033[0m" % (s, new_belief_state[d][s])
                    self.global_eval_stats[d][s]['TP'] += 1
                    is_printed = True
                if is_updated:
                    log_str += " (%s)" % (self.config.dst_class_types[state_updates[d][s]])
                    logging.info(log_str)
                elif is_printed:
                    logging.info(log_str)

    def eval_print_stats(self):
        logging.info("Statistics:")
        for d in self.global_eval_stats:
            for s in self.global_eval_stats[d]:
                TP = self.global_eval_stats[d][s]['TP']
                FP = self.global_eval_stats[d][s]['FP']
                FN = self.global_eval_stats[d][s]['FN']
                prec = TP / ( TP + FP + 1e-8)
                rec = TP / ( TP + FN + 1e-8)
                f1 = 2 * ((prec * rec) / (prec + rec + 1e-8))
                logging.info("  %s %s Recall: %.2f, Precision: %.2f, F1: %.2f" % (d, s, rec, prec, f1))

    def __init__(self, model_type="roberta",
                 model_name="roberta-base",
                 model_path="",
                 dataset_name="multiwoz21",
                 local_files_only=False,
                 nlu_usr_config="",
                 nlu_sys_config="",
                 nlu_usr_path="",
                 nlu_sys_path="",
                 no_eval=True,
                 no_history=False):
        super(TRIPPY, self).__init__()

        self.print_header()

        self.model_type = model_type.lower()
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.nlu_usr_config = nlu_usr_config
        self.nlu_sys_config = nlu_sys_config
        self.nlu_usr_path = nlu_usr_path
        self.nlu_sys_path = nlu_sys_path
        self.dataset_name = dataset_name
        self.no_eval = no_eval
        self.no_history = no_history

        assert self.model_type in ['roberta'] # TODO: ensure proper behavior for 'bert', 'electra'
        assert self.dataset_name in ['multiwoz21', 'multiwoz22', 'multiwoz23'] # TODO: ensure proper behavior for other datasets

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _ontology = relative_import_module_from_unified_datasets(self.dataset_name, 'preprocess.py', 'ontology')
        self.template_state = _ontology['state']
        
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_type]
        self.config = self.config_class.from_pretrained(self.model_path, local_files_only=self.local_files_only)

        self.dataset_interfacer = create_dataset_interfacer(dataset_name)

        # For internal evaluation only
        self.nlu_usr = None
        self.nlu_sys = None
        self.global_eval_stats = copy.deepcopy(self.template_state)
        for d in self.global_eval_stats:
            for s in self.global_eval_stats[d]:
                self.global_eval_stats[d][s] = {'TP': 0, 'FP': 0, 'FN': 0}
        self.global_diag_cnt = -3
        self.global_turn_cnt = -1
        if not self.no_eval:
            global BERTNLU
            from convlab.nlu.jointBERT.unified_datasets import BERTNLU
            self.load_nlu()

        # For semantic action pipelines only
        self.nlg_usr = None
        self.nlg_sys = None

        logging.info("DST INIT VARS: %s" % (vars(self)))

        self.load_weights()
    
    def load_weights(self):
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name, local_files_only=self.local_files_only) # TODO: do_lower_case?
        self.model = self.model_class.from_pretrained(self.model_path, config=self.config, local_files_only=self.local_files_only)
        self.model.to(self.device)
        self.model.eval()
        logging.info("DST model weights loaded from %s" % (self.model_path))

    def load_nlu(self):
        """ Loads NLUs for internal evaluation """
        # NLU for system utterances is used in case the policy does or can not provide semantic actions.
        # The sole purpose of this is to fill the inform memory.
        # NLU for user utterances is used in case the user simulator does or can not provide semantic actions.
        # The sole purpose of this is to enable internal DST evaluation.
        if self.nlu_usr_config == self.nlu_sys_config and \
           self.nlu_usr_path == self.nlu_sys_path:
            self.nlu_usr = BERTNLU(mode="all", config_file=self.nlu_usr_config, model_file=self.nlu_usr_path)
            self.nlu_sys = self.nlu_usr
        else:
            self.nlu_usr = BERTNLU(mode="user", config_file=self.nlu_usr_config, model_file=self.nlu_usr_path)
            self.nlu_sys = BERTNLU(mode="sys", config_file=self.nlu_sys_config, model_file=self.nlu_sys_path)
        logging.info("DST user NLU model weights loaded from %s" % (self.nlu_usr_path))
        logging.info("DST sys NLU model weights loaded from %s" % (self.nlu_sys_path))

    def load_nlg(self):
        if self.dataset_name in ['multiwoz21', 'multiwoz22', 'multiwoz23']:
            from convlab.nlg.template.multiwoz import TemplateNLG
            self.nlg_usr = TemplateNLG(is_user=True)
            self.nlg_sys = TemplateNLG(is_user=False)
            logging.info("DST template NLG loaded for dataset %s" % (self.dataset_name))
        else:
            raise Exception("DST no NLG for dataset %s available." % (self.dataset_name))
            
    def init_session(self):
        # Initialise empty state
        self.state = {'user_action': [],
                      'system_action': [],
                      'belief_state': {},
                      'booked': {},
                      'request_state': {},
                      'terminated': False,
                      'history': []}
        self.state['belief_state'] = copy.deepcopy(self.template_state)
        self.history = []
        self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        self.gt_belief_state = copy.deepcopy(self.template_state)
        self.global_diag_cnt += 1
        self.global_turn_cnt = -1

    def update_gt_belief_state(self, user_act):
        for intent, domain, slot, value in user_act:
            if domain == 'police':
                continue
            if intent == 'inform':
                if slot == 'none' or slot == '':
                    continue
                if slot in self.gt_belief_state[domain]:
                    self.gt_belief_state[domain][slot] = value

    def update(self, user_act=''):
        prev_state = self.state

        if not self.no_eval:
            logging.info("-" * 40)

        if self.no_history:
            self.history = []
        self.history.append(['sys', self.get_text(prev_state['history'][-2][1], is_user=False, normalize=True)])
        self.history.append(['user', self.get_text(prev_state['history'][-1][1], is_user=True, normalize=True)])

        self.global_turn_cnt += 1
        if not self.no_eval:
            self.print_dialog(self.history)

        # --- Get inform memory and auxiliary features ---

        # system_action is a list of semantic system actions.
        # TripPy uses system actions to fill the inform memory.
        # End-to-end policies like Lava produce plain text instead.
        # If system_action is plain text, get acts using NLU.
        if isinstance(prev_state['system_action'], str):
            s_acts = self.get_acts(prev_state['system_action'])
        elif isinstance(prev_state['system_action'], list):
            s_acts = prev_state['system_action']
        else:
            raise Exception('Unknown format for system action:', prev_state['system_action'])

        if not self.no_eval:
            # user_action is a list of semantic user actions if no NLG is used
            # in the pipeline, otherwise user_action is plain text.
            # TripPy uses user actions to perform internal DST evaluation.
            # If user_action is plain text, get acts using NLU.
            if isinstance(prev_state['user_action'], str):
                u_acts = self.get_acts(prev_state['user_action'], is_user=True)
            elif isinstance(prev_state['user_action'], list):
                u_acts = prev_state['user_action'] # This is the same as user_act
            else:
                raise Exception('Unknown format for user action:', prev_state['user_action'])

        # Fill the inform memory.
        inform_aux, inform_mem = self.get_inform_aux(s_acts)
        if not self.no_eval:
            self.print_inform_memory(inform_mem)

        # --- Tokenize dialogue context and feed DST model ---

        used_ds_aux = None if not self.config.dst_class_aux_feats_ds else self.ds_aux
        used_inform_aux = None if not self.config.dst_class_aux_feats_inform else inform_aux
        features = self.get_features(self.history, ds_aux=used_ds_aux, inform_aux=used_inform_aux)
        pred_states, pred_classes = self.predict(features, inform_mem)

        # --- Update ConvLab-style dialogue state ---

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        user_acts = []
        for state, value in pred_states.items():
            value = self.dataset_interfacer.normalize_values(value)
            domain, slot = state.split('-', 1)
            value = self.dataset_interfacer.normalize_prediction(domain, slot, value,
                                                                 predictions=pred_states,
                                                                 class_predictions=pred_classes,
                                                                 config=self.config)
            if value == 'none':
                continue
            if slot in new_belief_state[domain]:
                new_belief_state[domain][slot] = value
                user_acts.append(['inform', domain, slot, value])
            else:
                raise Exception('Unknown slot name <{}> with value <{}> of domain <{}>'.format(slot, value, domain))

        if not self.no_eval:
            self.update_gt_belief_state(u_acts) # For evaluation

        # BELIEF STATE UPDATE
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state # TripPy

        state_updates = {}
        for cl in pred_classes:
            cl_d, cl_s = cl.split('-')
            # Some reformatting for the evaluation further down
            if cl_d not in state_updates:
                state_updates[cl_d] = {}
            state_updates[cl_d][cl_s] = pred_classes[cl]
            # We care only about the requestable slots here
            if self.config.dst_class_types[pred_classes[cl]] != 'request':
                continue
            if cl_d != 'general' and cl_s == 'none':
                user_acts.append(['inform', cl_d, '', ''])
            elif cl_d == 'general':
                user_acts.append([cl_s, 'general', '', ''])
            else:
                user_acts.append(['request', cl_d, cl_s, ''])

        # USER ACTS UPDATE
        new_state['user_action'] = user_acts # TripPy

        if not self.no_eval:
            self.eval_user_acts(u_acts, user_acts)
            self.eval_dialog_state(state_updates, new_belief_state)

        self.state = new_state

        # Print eval statistics
        if self.state['terminated'] and not self.no_eval:
            logging.info("Booked: %s" % self.state['booked'])
            self.eval_print_stats()
            logging.info("=" * 10 + "End of the dialogue" + "=" * 10)
        self.ds_aux = self.update_ds_aux(self.state['belief_state'], pred_states)

        return self.state
    
    def predict(self, features, inform_mem):
        with torch.no_grad():
            outputs = self.model(input_ids=features['input_ids'],
                                 input_mask=features['attention_mask'],
                                 inform_slot_id=features['inform_slot_id'],
                                 diag_state=features['diag_state'])

        input_tokens = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0]) # unmasked!

        per_slot_class_logits = outputs[2]
        per_slot_start_logits = outputs[3]
        per_slot_end_logits = outputs[4]
        per_slot_refer_logits = outputs[5]

        # TODO: maybe add assert to check that batch=1
        
        predictions = {}
        class_predictions = {}

        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            slot_udf = "%s-%s" % (self.dataset_interfacer.map_trippy_to_udf(d, s))

            predictions[slot_udf] = 'none'
            class_predictions[slot_udf] = 0

            class_logits = per_slot_class_logits[slot][0]
            start_logits = per_slot_start_logits[slot][0]
            end_logits = per_slot_end_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if class_prediction == self.config.dst_class_types.index('dontcare'):
                predictions[slot_udf] = 'dontcare'
            elif class_prediction == self.config.dst_class_types.index('copy_value'):
                predictions[slot_udf] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                predictions[slot_udf] = re.sub("(^| )##", "", predictions[slot_udf])
                if "\u0120" in predictions[slot_udf]:
                    predictions[slot_udf] = re.sub(" ", "", predictions[slot_udf])
                    predictions[slot_udf] = re.sub("\u0120", " ", predictions[slot_udf])
                    predictions[slot_udf] = predictions[slot_udf].strip()
            elif 'true' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('true'):
                predictions[slot_udf] = "yes" # 'true'
            elif 'false' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('false'):
                predictions[slot_udf] = "no" # 'false'
            elif class_prediction == self.config.dst_class_types.index('inform'):
                predictions[slot_udf] = inform_mem[slot_udf]
            # Referral case is handled below

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            slot_udf = "%s-%s" % (self.dataset_interfacer.map_trippy_to_udf(d, s))

            class_logits = per_slot_class_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # First try to resolve a reference within the same turn. (One can think of a situation
                # where one slot is referred to in the same utterance. This phenomenon is however
                # currently not properly covered in the training data label generation process)
                # Then try to resolve a reference given the current dialogue state.
                referred_slot = self.config.dst_slot_list[refer_prediction - 1]
                referred_slot_d, referred_slot_s = referred_slot.split('-')
                referred_slot_d, referred_slot_s = self.dataset_interfacer.map_trippy_to_udf(referred_slot_d, referred_slot_s)
                referred_slot_udf = "%s-%s" % (referred_slot_d, referred_slot_s)
                predictions[slot_udf] = predictions[referred_slot_udf]
                if predictions[slot_udf] == 'none':
                    if self.state['belief_state'][referred_slot_d][referred_slot_s] != '':
                        predictions[slot_udf] = self.state['belief_state'][referred_slot_d][referred_slot_s]
                if predictions[slot_udf] == 'none':
                    ref_slot = self.config.dst_slot_list[refer_prediction - 1]
                    ref_slot_d, ref_slot_s = ref_slot.split('-')
                    generic_ref = self.dataset_interfacer.get_generic_referral(ref_slot_d, ref_slot_s)
                    predictions[slot_udf] = generic_ref

            class_predictions[slot_udf] = class_prediction

        return predictions, class_predictions

    def get_features(self, context, ds_aux=None, inform_aux=None):
        assert(self.model_type == "roberta") # TODO: generalize to other BERT-like models
        input_tokens = ['<s>'] # TODO: use tokenizer token names rather than strings
        e_itr = 0
        for e_itr, e in enumerate(reversed(context)):
            if e[1] not in ['null', '']:
                input_tokens.append(e[1])
            if e_itr < 2:
                input_tokens.append('</s> </s>')
        if e_itr == 0:
            input_tokens.append('</s> </s>')
        input_tokens.append('</s>')
        input_tokens = ' '.join(input_tokens)

        # TODO: delex sys utt currently not supported
        features = self.tokenizer.encode_plus(input_tokens, add_special_tokens=False, max_length=self.config.dst_max_seq_length)

        input_ids = torch.tensor(features['input_ids']).reshape(1,-1).to(self.device)
        attention_mask = torch.tensor(features['attention_mask']).reshape(1,-1).to(self.device)
        features = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'inform_slot_id': inform_aux,
                    'diag_state': ds_aux}

        return features

    def update_ds_aux(self, state, pred_states, terminated=False):
        ds_aux = copy.deepcopy(self.ds_aux)
        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            d_udf, s_udf = self.dataset_interfacer.map_trippy_to_udf(d, s)
            slot_udf = "%s-%s" % (d_udf, s_udf)
            if d_udf in state and s_udf in state[d_udf]:
                ds_aux[slot][0] = int(state[d_udf][s_udf] != '')
            else:
                # Requestable slots are not found in the DS
                ds_aux[slot][0] = int(pred_states[slot_udf] != 'none')
        return ds_aux

    def get_inform_aux(self, state):
        # Initialise auxiliary variables.
        # For inform_aux, only the proper order of slots
        # as defined in dst_slot_list is relevant, but not
        # the actual slot names (as inform_aux will be
        # converted into a simple binary list in the model)
        inform_aux = {}
        inform_mem = {}
        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            d_udf, s_udf = self.dataset_interfacer.map_trippy_to_udf(d, s)
            inform_aux["%s-%s" % (d_udf, s_udf)] = torch.tensor([0]).to(self.device)
            inform_mem["%s-%s" % (d_udf, s_udf)] = 'none'
        for e in state:
            a, d, s, v = e
            # TODO: offerbook needed? booked needed?
            if a in ['inform', 'recommend', 'select', 'book', 'offerbook']:
                slot = "%s-%s" % (d, s)
                if slot in inform_aux:
                    inform_aux[slot][0] = 1
                    inform_mem[slot] = self.dataset_interfacer.normalize_values(v)
        return inform_aux, inform_mem

    def get_acts(self, act, is_user=False):
        if isinstance(act, list):
            return act
        context = self.state['history']
        if context[-1][0] not in ['user', 'usr']:
            raise Exception("Wrong order of utterances, check your input.")
        system_context = [self.get_text(t) for s,t in context[:-2]]
        user_context = [self.get_text(t, is_user=True) for s,t in context[:-1]]
        if is_user:
            if self.nlu_usr is None:
                raise Exception("You attempt to convert semantic user actions into text, but no NLU module is loaded.")
            acts = self.nlu_usr.predict(act, context=user_context)
        else:
            if self.nlu_sys is None:
                raise Exception("You attempt to convert semantic system actions into text, but no NLU module is loaded.")
            acts = self.nlu_sys.predict(act, context=system_context)
        for act_itr in range(len(acts)):
            acts[act_itr][-1] = self.dataset_interfacer.normalize_values(acts[act_itr][-1])
        return acts

    def get_text(self, act, is_user=False, normalize=False):
        if act == 'null':
            return 'null'
        if not isinstance(act, list):
            result = act
        else:
            if self.nlg_usr is None or self.nlg_sys is None:
                logging.warn("You attempt to input semantic actions into TripPy, which expects text.")
                logging.warn("Attempting to load NLG modules in order to convert actions into text.")
                self.load_nlg()
            if is_user:
                result = self.nlg_usr.generate(act)
            else:
                result = self.nlg_sys.generate(act)
        if normalize:
            return self.dataset_interfacer.normalize_text(result)
        else:
            return result
