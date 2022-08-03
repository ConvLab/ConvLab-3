# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Ensemble SetSUMBT"""

import os

import torch
import transformers
from torch.nn import Module
from transformers import RobertaConfig, BertConfig

from convlab.dst.setsumbt.modeling.bert_nbt import BertSetSUMBT
from convlab.dst.setsumbt.modeling.roberta_nbt import RobertaSetSUMBT

MODELS = {'bert': BertSetSUMBT, 'roberta': RobertaSetSUMBT}


class EnsembleSetSUMBT(Module):

    def __init__(self, config):
        super(EnsembleSetSUMBT, self).__init__()
        self.config = config

        # Initialise ensemble members
        model_cls = MODELS[self.config.model_type]
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            setattr(self, attr, model_cls(config))
    

    # Load all ensemble member parameters
    def load(self, path, config=None):
        if config is None:
            config = self.config
        
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            idx = attr.split('_', 1)[-1]
            state_dict = torch.load(os.path.join(path, f'pytorch_model_{idx}.bin'))
            getattr(self, attr).load_state_dict(state_dict)
    

    # Add new slot candidates to the ensemble members
    def add_slot_candidates(self, slot_candidates):
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            getattr(self, attr).add_slot_candidates(slot_candidates)
        self.requestable_slot_ids = self.model_0.requestable_slot_ids
        self.informable_slot_ids = self.model_0.informable_slot_ids
        self.domain_ids = self.model_0.domain_ids


    # Add new value candidates to the ensemble members
    def add_value_candidates(self, slot, value_candidates, replace=False):
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            getattr(self, attr).add_value_candidates(slot, value_candidates, replace)
        

    # Forward pass of full ensemble
    def forward(self, input_ids, attention_mask, token_type_ids=None, reduction='mean'):
        logits, request_logits, domain_logits, goodbye_scores = [], [], [], []
        logits = {slot: [] for slot in self.model_0.informable_slot_ids}
        request_logits = {slot: [] for slot in self.model_0.requestable_slot_ids}
        domain_logits = {dom: [] for dom in self.model_0.domain_ids}
        goodbye_scores = []
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            # Prediction from each ensemble member
            l, r, d, g, _ = getattr(self, attr)(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
            for slot in logits:
                logits[slot].append(l[slot].unsqueeze(-2))
            if self.config.predict_intents:
                for slot in request_logits:
                    request_logits[slot].append(r[slot].unsqueeze(-1))
                for dom in domain_logits:
                    domain_logits[dom].append(d[dom].unsqueeze(-1))
                goodbye_scores.append(g.unsqueeze(-2))
        
        logits = {slot: torch.cat(l, -2) for slot, l in logits.items()}
        if self.config.predict_intents:
            request_logits = {slot: torch.cat(l, -1) for slot, l in request_logits.items()}
            domain_logits = {dom: torch.cat(l, -1) for dom, l in domain_logits.items()}
            goodbye_scores = torch.cat(goodbye_scores, -2)
        else:
            request_logits = {}
            domain_logits = {}
            goodbye_scores = torch.tensor(0.0)

        # Apply reduction of ensemble to single posterior
        if reduction == 'mean':
            logits = {slot: l.mean(-2) for slot, l in logits.items()}
            request_logits = {slot: l.mean(-1) for slot, l in request_logits.items()}
            domain_logits = {dom: l.mean(-1) for dom, l in domain_logits.items()}
            goodbye_scores = goodbye_scores.mean(-2)
        elif reduction != 'none':
            raise(NameError('Not Implemented!'))

        return logits, request_logits, domain_logits, goodbye_scores, _
    

    @classmethod
    def from_pretrained(cls, path):
        if not os.path.exists(os.path.join(path, 'config.json')):
            raise(NameError('Could not find config.json in model path.'))
        if not os.path.exists(os.path.join(path, 'pytorch_model_0.bin')):
            raise(NameError('Could not find a model binary in the model path.'))
        
        try:
            config = RobertaConfig.from_pretrained(path)
        except:
            config = BertConfig.from_pretrained(path)
        
        model = cls(config)
        model.load(path)

        return model


class DropoutEnsembleSetSUMBT(Module):

    def __init__(self, config):
        super(DropoutEnsembleBeliefTracker, self).__init__()
        self.config = config

        model_cls = MODELS[self.config.model_type]
        self.model = model_cls(config)
        self.model.train()
    

    def load(self, path, config=None):
        if config is None:
            config = self.config
        state_dict = torch.load(os.path.join(path, f'pytorch_model.bin'))
        self.model.load_state_dict(state_dict)
    

    # Add new slot candidates to the model
    def add_slot_candidates(self, slot_candidates):
        self.model.add_slot_candidates(slot_candidates)
        self.requestable_slot_ids = self.model.requestable_slot_ids
        self.informable_slot_ids = self.model.informable_slot_ids
        self.domain_ids = self.model.domain_ids


    # Add new value candidates to the model
    def add_value_candidates(self, slot, value_candidates, replace=False):
        self.model.add_value_candidates(slot, value_candidates, replace)
        
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, reduction='mean'):

        input_ids = input_ids.unsqueeze(0).repeat((self.config.ensemble_size, 1, 1, 1))
        input_ids = input_ids.reshape(-1, input_ids.size(-2), input_ids.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).repeat((10, 1, 1, 1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-2), attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(0).repeat((10, 1, 1, 1))
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-2), token_type_ids.size(-1))
        
        self.model.train()
        logits, request_logits, domain_logits, goodbye_scores, _ = self.model(input_ids=input_ids,
                                                                            attention_mask=attention_mask,
                                                                            token_type_ids=token_type_ids)
        
        logits = {s: l.reshape(self.config.ensemble_size, -1, l.size(-2), l.size(-1)).transpose(0, 1).transpose(1, 2)
                for s, l in logits.items()}
        request_logits = {s: l.reshape(self.config.ensemble_size, -1, l.size(-1)).transpose(0, 1).transpose(1, 2)
                        for s, l in request_logits.items()}
        domain_logits = {s: l.reshape(self.config.ensemble_size, -1, l.size(-1)).transpose(0, 1).transpose(1, 2)
                        for s, l in domain_logits.items()}
        goodbye_scores = goodbye_scores.reshape(self.config.ensemble_size, -1, goodbye_scores.size(-2), goodbye_scores.size(-1))
        goodbye_scores = goodbye_scores.transpose(0, 1).transpose(1, 2)

        if reduction == 'mean':
            logits = {slot: l.mean(-2) for slot, l in logits.items()}
            request_logits = {slot: l.mean(-1) for slot, l in request_logits.items()}
            domain_logits = {dom: l.mean(-1) for dom, l in domain_logits.items()}
            goodbye_scores = goodbye_scores.mean(-2)
        elif reduction != 'none':
            raise(NameError('Not Implemented!'))

        return logits, request_logits, domain_logits, goodbye_scores, _
    

    @classmethod
    def from_pretrained(cls, path):
        if not os.path.exists(os.path.join(path, 'config.json')):
            raise(NameError('Could not find config.json in model path.'))
        if not os.path.exists(os.path.join(path, 'pytorch_model.bin')):
            raise(NameError('Could not find a model binary in the model path.'))
        
        try:
            config = RobertaConfig.from_pretrained(path)
        except:
            config = BertConfig.from_pretrained(path)
        
        model = cls(config)
        model.load(path)

        return model
