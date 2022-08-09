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
from torch.nn import Module
from transformers import RobertaConfig, BertConfig

from convlab.dst.setsumbt.modeling.bert_nbt import BertSetSUMBT
from convlab.dst.setsumbt.modeling.roberta_nbt import RobertaSetSUMBT

MODELS = {'bert': BertSetSUMBT, 'roberta': RobertaSetSUMBT}


class EnsembleSetSUMBT(Module):
    """Ensemble SetSUMBT Model for joint ensemble prediction"""

    def __init__(self, config):
        """
        Args:
            config (configuration): Model configuration class
        """
        super(EnsembleSetSUMBT, self).__init__()
        self.config = config

        # Initialise ensemble members
        model_cls = MODELS[self.config.model_type]
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            setattr(self, attr, model_cls(config))

    def _load(self, path: str):
        """
        Load parameters
        Args:
            path: Location of model parameters
        """
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            idx = attr.split('_', 1)[-1]
            state_dict = torch.load(os.path.join(path, f'pytorch_model_{idx}.bin'))
            getattr(self, attr).load_state_dict(state_dict)

    def add_slot_candidates(self, slot_candidates: tuple):
        """
        Add slots to the model ontology, the tuples should contain the slot embedding, informable value embeddings
        and a request indicator, if the informable value embeddings is None the slot is not informable and if
        the request indicator is false the slot is not requestable.

        Args:
            slot_candidates: Tuple containing slot embedding, informable value embeddings and a request indicator
        """
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            getattr(self, attr).add_slot_candidates(slot_candidates)
        self.requestable_slot_ids = self.model_0.setsumbt.requestable_slot_ids
        self.informable_slot_ids = self.model_0.setsumbt.informable_slot_ids
        self.domain_ids = self.setsumbt.model_0.domain_ids

    def add_value_candidates(self, slot: str, value_candidates: torch.Tensor, replace: bool = False):
        """
        Add value candidates for a slot

        Args:
            slot: Slot name
            value_candidates: Value candidate embeddings
            replace: If true existing value candidates are replaced
        """
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            getattr(self, attr).add_value_candidates(slot, value_candidates, replace)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                reduction: str = 'mean') -> tuple:
        """
        Args:
            input_ids: Input token ids
            attention_mask: Input padding mask
            token_type_ids: Token type indicator
            reduction: Reduction of ensemble member predictive distributions (mean, none)

        Returns:

        """
        belief_state_probs = {slot: [] for slot in self.model_0.informable_slot_ids}
        request_probs = {slot: [] for slot in self.model_0.requestable_slot_ids}
        active_domain_probs = {dom: [] for dom in self.model_0.domain_ids}
        general_act_probs = []
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            # Prediction from each ensemble member
            b, r, d, g, _ = getattr(self, attr)(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
            for slot in belief_state_probs:
                belief_state_probs[slot].append(b[slot].unsqueeze(-2))
            if self.config.predict_intents:
                for slot in request_probs:
                    request_probs[slot].append(r[slot].unsqueeze(-1))
                for dom in active_domain_probs:
                    active_domain_probs[dom].append(d[dom].unsqueeze(-1))
                general_act_probs.append(g.unsqueeze(-2))
        
        belief_state_probs = {slot: torch.cat(l, -2) for slot, l in belief_state_probs.items()}
        if self.config.predict_intents:
            request_probs = {slot: torch.cat(l, -1) for slot, l in request_probs.items()}
            active_domain_probs = {dom: torch.cat(l, -1) for dom, l in active_domain_probs.items()}
            general_act_probs = torch.cat(general_act_probs, -2)
        else:
            request_probs = {}
            active_domain_probs = {}
            general_act_probs = torch.tensor(0.0)

        # Apply reduction of ensemble to single posterior
        if reduction == 'mean':
            belief_state_probs = {slot: l.mean(-2) for slot, l in belief_state_probs.items()}
            request_probs = {slot: l.mean(-1) for slot, l in request_probs.items()}
            active_domain_probs = {dom: l.mean(-1) for dom, l in active_domain_probs.items()}
            general_act_probs = general_act_probs.mean(-2)
        elif reduction != 'none':
            raise(NameError('Not Implemented!'))

        return belief_state_probs, request_probs, active_domain_probs, general_act_probs, _
    

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
