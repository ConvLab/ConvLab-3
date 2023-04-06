# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
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
"""SetSUMBT Tokenizer"""

import json
import os

import torch
from transformers import RobertaTokenizer, BertTokenizer
from tqdm import tqdm

from convlab.dst.setsumbt.datasets.utils import IdTensor

PARENT_CLASSES = {'bert': BertTokenizer,
                  'roberta': RobertaTokenizer}


def SetSUMBTTokenizer(parent_name):
    """SetSUMBT Tokenizer Class Factory"""
    parent_class = PARENT_CLASSES.get(parent_name.lower())

    class SetSUMBTTokenizer(parent_class):
        """SetSUMBT Tokenizer Class"""

        def __init__(
                self,
                vocab_file,
                merges_file,
                errors="replace",
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                add_prefix_space=False,
                **kwargs,
        ):
            """
            Initialize the tokenizer.

            Args:
                vocab_file (str): Path to the vocabulary file.
                merges_file (str): Path to the merges file.
                errors (str): Error handling for the tokenizer.
                bos_token (str): Beginning of sentence token.
                eos_token (str): End of sentence token.
                sep_token (str): Separator token.
                cls_token (str): Classification token.
                unk_token (str): Unknown token.
                pad_token (str): Padding token.
                mask_token (str): Masking token.
                add_prefix_space (bool): Whether to add a space before the first token.
                **kwargs: Additional arguments for the tokenizer.
            """

            # Load ontology and tokenizer vocab
            with open(vocab_file, 'r', encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
                vocab_handle.close()
            self.ontology = self.encoder['SETSUMBT_ONTOLOGY'] if 'SETSUMBT_ONTOLOGY' in self.encoder else dict()
            self.encoder = {k: v for k, v in self.encoder.items() if 'SETSUMBT_ONTOLOGY' not in k}
            vocab_dir = os.path.dirname(vocab_file)
            vocab_file = os.path.basename(vocab_file).split('.')
            vocab_file = vocab_file[0] + "_base." + vocab_file[-1]
            vocab_file = os.path.join(vocab_dir, vocab_file)
            with open(vocab_file, 'w', encoding="utf-8") as vocab_handle:
                json.dump(self.encoder, vocab_handle)
                vocab_handle.close()

            super().__init__(vocab_file, merges_file, errors, bos_token, eos_token, sep_token, cls_token, unk_token,
                             pad_token, mask_token, add_prefix_space, **kwargs)

        def set_setsumbt_ontology(self, ontology):
            """
            Set the ontology for the tokenizer.

            Args:
                ontology (dict): The dialogue system ontology to use.
            """
            self.ontology = ontology

        def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
            """
            Save the tokenizer vocabulary and merges files to a directory.

            Args:
                save_directory (str): Directory to which to save.
                filename_prefix (str): Optional prefix to add to the files.

            Returns:
                vocab_file (str): Path to the saved vocabulary file.
                merge_file (str): Path to the saved merges file.
            """
            self.encoder['SETSUMBT_ONTOLOGY'] = self.ontology
            vocab_file, merge_file = super().save_vocabulary(save_directory, filename_prefix)
            self.encoder = {k: v for k, v in self.encoder.items() if 'SETSUMBT_ONTOLOGY' not in k}

            return vocab_file, merge_file

        def decode_state(self, belief_state, request_probs=None, active_domain_probs=None, general_act_probs=None):
            """
            Decode a belief state, request, active domain and general action distributions into a dialogue state.

            Args:
                belief_state (dict): The belief state distributions.
                request_probs (dict): The request distributions.
                active_domain_probs (dict): The active domain distributions.
                general_act_probs (dict): The general action distributions.

            Returns:
                dialogue_state (dict): The decoded dialogue state.
            """
            dialogue_state = {domain: {slot: '' for slot, slot_info in domain_info.items()
                                       if slot_info['possible_values'] != ["?"] and slot_info['possible_values']}
                              for domain, domain_info in self.ontology.items()}

            for slot, probs in belief_state.items():
                dom, slot = slot.split('-', 1)
                val = self.ontology.get(dom, dict()).get(slot, dict()).get('possible_values', [])
                val = val[probs.argmax().item()] if val else 'none'
                if val != 'none':
                    if dom in dialogue_state:
                        if slot in dialogue_state[dom]:
                            dialogue_state[dom][slot] = val

            request_acts = list()
            if request_probs is not None:
                request_acts = [slot for slot, p in request_probs.items() if p.item() > 0.5]
                request_acts = [slot.split('-', 1) for slot in request_acts]
                request_acts = [[dom, slt] for dom, slt in request_acts
                                if '?' in self.ontology.get(dom, dict()).get(slt, dict()).get('possible_values', [])]
                request_acts = [['request', domain, slot, '?'] for domain, slot in request_acts]

            # Construct active domain set
            active_domains = dict()
            if active_domain_probs is not None:
                active_domains = {dom: active_domain_probs.get(dom, torch.tensor(0.0)).item() > 0.5
                                  for dom in self.ontology}

            # Construct general domain action
            general_acts = list()
            if general_act_probs is not None:
                general_acts = general_act_probs.argmax(-1).item()
                general_acts = [[], ['bye'], ['thank']][general_acts]
                general_acts = [[act, 'general', 'none', 'none'] for act in general_acts]

            user_acts = request_acts + general_acts
            dialogue_state = {'belief_state': dialogue_state,
                              'user_action': user_acts,
                              'active_domains': active_domains}

            return dialogue_state

        def decode_state_batch(self,
                               belief_state,
                               request_probs=None,
                               active_domain_probs=None,
                               general_act_probs=None,
                               dialogue_ids=None):
            """
            Decode a batch of belief state, request, active domain and general action distributions.

            Args:
                belief_state (dict): The belief state distributions.
                request_probs (dict): The request distributions.
                active_domain_probs (dict): The active domain distributions.
                general_act_probs (dict): The general action distributions.
                dialogue_ids (list): The dialogue IDs.

            Returns:
                data (dict): The decoded dialogue states.
            """

            data = dict()
            slot_0 = [key for key in belief_state.keys()][0]

            if dialogue_ids is None:
                dialogue_ids = [["{:06d}".format(i) for i in range(belief_state[slot_0].size(0))]]

            for dial_idx in range(belief_state[slot_0].size(0)):
                dialogue = list()
                for turn_idx in range(belief_state[slot_0].size(1)):
                    if belief_state[slot_0][dial_idx, turn_idx].sum() != 0.0:
                        belief = {slot: p[dial_idx, turn_idx] for slot, p in belief_state.items()}
                        req = {slot: p[dial_idx, turn_idx]
                               for slot, p in request_probs.items()} if request_probs is not None else None
                        dom = {dom: p[dial_idx, turn_idx]
                               for dom, p in active_domain_probs.items()} if active_domain_probs is not None else None
                        gen = general_act_probs[dial_idx, turn_idx] if general_act_probs is not None else None

                        state = self.decode_state(belief, req, dom, gen)
                        dialogue.append(state)
                data[dialogue_ids[0][dial_idx]] = dialogue

            return data

        def encode(self, dialogues: list, max_turns: int = 12, max_seq_len: int = 64) -> dict:
            """
            Convert dialogue examples to model input features and labels

            Args:
                dialogues (list): List of all extracted dialogues
                max_turns (int): Maximum numbers of turns in a dialogue
                max_seq_len (int): Maximum number of tokens in a dialogue turn

            Returns:
                features (dict): All inputs and labels required to train the model
            """
            features = dict()

            # Get encoder input for system, user utterance pairs
            input_feats = []
            if len(dialogues) > 5:
                iterator = tqdm(dialogues)
            else:
                iterator = dialogues
            for dial in iterator:
                dial_feats = []
                for turn in dial:
                    if len(turn['system_utterance']) == 0:
                        usr = turn['user_utterance']
                        dial_feats.append(super().encode_plus(usr, add_special_tokens=True, max_length=max_seq_len,
                                                              padding='max_length', truncation='longest_first'))
                    else:
                        usr = turn['user_utterance']
                        sys = turn['system_utterance']
                        dial_feats.append(super().encode_plus(usr, sys, add_special_tokens=True,
                                                              max_length=max_seq_len, padding='max_length',
                                                              truncation='longest_first'))
                    # Truncate
                    if len(dial_feats) >= max_turns:
                        break
                input_feats.append(dial_feats)
            del dial_feats

            # Perform turn level padding
            if 'dialogue_id' in dialogues[0][0]:
                dial_ids = list()
                for dial in dialogues:
                    _ids = [turn['dialogue_id'] for turn in dial][:max_turns]
                    _ids += [''] * (max_turns - len(_ids))
                    dial_ids.append(_ids)
            input_ids = [[turn['input_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                         for dial in input_feats]
            if 'token_type_ids' in input_feats[0][0]:
                token_type_ids = [[turn['token_type_ids'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                                  for dial in input_feats]
            else:
                token_type_ids = None
            if 'attention_mask' in input_feats[0][0]:
                attention_mask = [[turn['attention_mask'] for turn in dial] + [[0] * max_seq_len] * (max_turns - len(dial))
                                  for dial in input_feats]
            else:
                attention_mask = None
            del input_feats

            # Create torch data tensors
            if 'dialogue_id' in dialogues[0][0]:
                features['dialogue_ids'] = IdTensor(dial_ids)
            features['input_ids'] = torch.tensor(input_ids)
            features['token_type_ids'] = torch.tensor(token_type_ids) if token_type_ids else None
            features['attention_mask'] = torch.tensor(attention_mask) if attention_mask else None
            del input_ids, token_type_ids, attention_mask

            # Extract all informable and requestable slots from the ontology
            informable_slots = [f"{domain}-{slot}" for domain in self.ontology for slot in self.ontology[domain]
                                if self.ontology[domain][slot]['possible_values']
                                and self.ontology[domain][slot]['possible_values'] != ['?']]
            requestable_slots = [f"{domain}-{slot}" for domain in self.ontology for slot in self.ontology[domain]
                                 if '?' in self.ontology[domain][slot]['possible_values']]

            # Extract a list of domains from the ontology slots
            domains = [domain for domain in self.ontology]

            # Create slot labels
            if 'state' in dialogues[0][0]:
                for domslot in tqdm(informable_slots):
                    labels = []
                    for dial in dialogues:
                        labs = []
                        for turn in dial:
                            value = [v for d, substate in turn['state'].items() for s, v in substate.items()
                                     if f'{d}-{s}' == domslot]
                            domain, slot = domslot.split('-', 1)
                            if turn['dataset_name'] in self.ontology[domain][slot]['dataset_names']:
                                value = value[0] if value else 'none'
                            else:
                                value = -1
                            if value in self.ontology[domain][slot]['possible_values'] and value != -1:
                                value = self.ontology[domain][slot]['possible_values'].index(value)
                            else:
                                value = -1  # If value is not in ontology then we do not penalise the model
                            labs.append(value)
                            if len(labs) >= max_turns:
                                break
                        labs = labs + [-1] * (max_turns - len(labs))
                        labels.append(labs)

                    labels = torch.tensor(labels)
                    features['state_labels-' + domslot] = labels

            # Create requestable slot labels
            if 'dialogue_acts' in dialogues[0][0]:
                for domslot in tqdm(requestable_slots):
                    labels = []
                    for dial in dialogues:
                        labs = []
                        for turn in dial:
                            domain, slot = domslot.split('-', 1)
                            if turn['dataset_name'] in self.ontology[domain][slot]['dataset_names']:
                                acts = [act['intent'] for act in turn['dialogue_acts']
                                        if act['domain'] == domain and act['slot'] == slot]
                                if acts:
                                    act_ = acts[0]
                                    if act_ == 'request':
                                        labs.append(1)
                                    else:
                                        labs.append(0)
                                else:
                                    labs.append(0)
                            else:
                                labs.append(-1)
                            if len(labs) >= max_turns:
                                break
                        labs = labs + [-1] * (max_turns - len(labs))
                        labels.append(labs)

                    labels = torch.tensor(labels)
                    features['request_labels-' + domslot] = labels

                # General act labels (1-goodbye, 2-thank you)
                labels = []
                for dial in tqdm(dialogues):
                    labs = []
                    for turn in dial:
                        acts = [act['intent'] for act in turn['dialogue_acts'] if act['intent'] in ['bye', 'thank']]
                        if acts:
                            if 'bye' in acts:
                                labs.append(1)
                            else:
                                labs.append(2)
                        else:
                            labs.append(0)
                        if len(labs) >= max_turns:
                            break
                    labs = labs + [-1] * (max_turns - len(labs))
                    labels.append(labs)

                labels = torch.tensor(labels)
                features['general_act_labels'] = labels

            # Create active domain labels
            if 'active_domains' in dialogues[0][0]:
                for domain in tqdm(domains):
                    labels = []
                    for dial in dialogues:
                        labs = []
                        for turn in dial:
                            possible_domains = list()
                            for dom in self.ontology:
                                for slt in self.ontology[dom]:
                                    if turn['dataset_name'] in self.ontology[dom][slt]['dataset_names']:
                                        possible_domains.append(dom)

                            if domain in turn['active_domains']:
                                labs.append(1)
                            elif domain in possible_domains:
                                labs.append(0)
                            else:
                                labs.append(-1)
                            if len(labs) >= max_turns:
                                break
                        labs = labs + [-1] * (max_turns - len(labs))
                        labels.append(labs)

                    labels = torch.tensor(labels)
                    features['active_domain_labels-' + domain] = labels

            try:
                del labels
            except:
                labels = None

            return features

    return SetSUMBTTokenizer
