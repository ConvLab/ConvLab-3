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
"""SetSUMBT Models"""

import os
from copy import deepcopy

import torch
from torch.nn import Module
from transformers import (BertModel, BertPreTrainedModel, BertConfig,
                          RobertaModel, RobertaPreTrainedModel, RobertaConfig)

from convlab.dst.setsumbt.modeling.setsumbt import SetSUMBTHead, SetSUMBTOutput


class BertSetSUMBT(BertPreTrainedModel):
    """Bert based SetSUMBT model"""

    def __init__(self, config):
        """
        Args:
            config (configuration): Model configuration class
        """
        super(BertSetSUMBT, self).__init__(config)
        self.config = config

        # Turn Encoder
        self.bert = BertModel(config)
        if config.freeze_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.setsumbt = SetSUMBTHead(config)
        self.add_slot_candidates = self.setsumbt.add_slot_candidates
        self.add_value_candidates = self.setsumbt.add_value_candidates

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                hidden_state: torch.Tensor = None,
                state_labels: torch.Tensor = None,
                request_labels: torch.Tensor = None,
                active_domain_labels: torch.Tensor = None,
                general_act_labels: torch.Tensor = None,
                get_turn_pooled_representation: bool = False,
                calculate_state_mutual_info: bool = False):
        """
        Args:
            input_ids: Input token ids
            attention_mask: Input padding mask
            token_type_ids: Token type indicator
            hidden_state: Latent internal dialogue belief state
            state_labels: Dialogue state labels
            request_labels: User request action labels
            active_domain_labels: Current active domain labels
            general_act_labels: General user action labels
            get_turn_pooled_representation: Return pooled representation of the current dialogue turn
            calculate_state_mutual_info: Return mutual information in the dialogue state

        Returns:
            out: Tuple containing loss, predictive distributions, model statistics and state mutual information
        """

        # Encode Dialogues
        batch_size, dialogue_size, turn_size = input_ids.size()
        input_ids = input_ids.reshape(-1, turn_size)
        token_type_ids = token_type_ids.reshape(-1, turn_size)
        attention_mask = attention_mask.reshape(-1, turn_size)

        bert_output = self.bert(input_ids, token_type_ids, attention_mask)

        attention_mask = attention_mask.float().unsqueeze(2)
        attention_mask = attention_mask.repeat((1, 1, bert_output.last_hidden_state.size(-1)))
        turn_embeddings = bert_output.last_hidden_state * attention_mask
        turn_embeddings = turn_embeddings.reshape(batch_size * dialogue_size, turn_size, -1)

        output = self.setsumbt(turn_embeddings, bert_output.pooler_output, attention_mask,
                               batch_size, dialogue_size, hidden_state, state_labels,
                               request_labels, active_domain_labels, general_act_labels,
                               calculate_state_mutual_info)
        output.turn_pooled_representation = bert_output.pooler_output if get_turn_pooled_representation else None
        return output


class RobertaSetSUMBT(RobertaPreTrainedModel):
    """Roberta based SetSUMBT model"""

    def __init__(self, config):
        """
        Args:
            config (configuration): Model configuration class
        """
        super(RobertaSetSUMBT, self).__init__(config)
        self.config = config

        # Turn Encoder
        self.roberta = RobertaModel(config)
        if config.freeze_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.setsumbt = SetSUMBTHead(config)
        self.add_slot_candidates = self.setsumbt.add_slot_candidates
        self.add_value_candidates = self.setsumbt.add_value_candidates

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                hidden_state: torch.Tensor = None,
                state_labels: torch.Tensor = None,
                request_labels: torch.Tensor = None,
                active_domain_labels: torch.Tensor = None,
                general_act_labels: torch.Tensor = None,
                get_turn_pooled_representation: bool = False,
                calculate_state_mutual_info: bool = False):
        """
        Args:
            input_ids: Input token ids
            attention_mask: Input padding mask
            token_type_ids: Token type indicator
            hidden_state: Latent internal dialogue belief state
            state_labels: Dialogue state labels
            request_labels: User request action labels
            active_domain_labels: Current active domain labels
            general_act_labels: General user action labels
            get_turn_pooled_representation: Return pooled representation of the current dialogue turn
            calculate_state_mutual_info: Return mutual information in the dialogue state

        Returns:
            out: Tuple containing loss, predictive distributions, model statistics and state mutual information
        """
        if token_type_ids is not None:
            token_type_ids = None

        # Encode Dialogues
        batch_size, dialogue_size, turn_size = input_ids.size()
        input_ids = input_ids.reshape(-1, turn_size)
        attention_mask = attention_mask.reshape(-1, turn_size)

        roberta_output = self.roberta(input_ids, attention_mask)

        # Apply mask and reshape the dialogue turn token embeddings
        attention_mask = attention_mask.float().unsqueeze(2)
        attention_mask = attention_mask.repeat((1, 1, roberta_output.last_hidden_state.size(-1)))
        turn_embeddings = roberta_output.last_hidden_state * attention_mask
        turn_embeddings = turn_embeddings.reshape(batch_size * dialogue_size, turn_size, -1)

        output = self.setsumbt(turn_embeddings, roberta_output.pooler_output, attention_mask,
                               batch_size, dialogue_size, hidden_state, state_labels,
                               request_labels, active_domain_labels, general_act_labels,
                               calculate_state_mutual_info)
        output.turn_pooled_representation = roberta_output.pooler_output if get_turn_pooled_representation else None
        return output


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
            setattr(self, attr, model_cls(self.get_clean_config(config)))

    @staticmethod
    def get_clean_config(config):
        config = deepcopy(config)
        config.slot_ids = dict()
        config.requestable_slot_ids = dict()
        config.informable_slot_ids = dict()
        config.domain_ids = dict()
        config.num_values = dict()

        return config

    def _load(self, path: str):
        """
        Load parameters
        Args:
            path: Location of model parameters
        """
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            idx = attr.split('_', 1)[-1]
            state_dict = torch.load(os.path.join(self._get_checkpoint_path(path, idx), 'pytorch_model.bin'))
            state_dict = {key: itm for key, itm in state_dict.items() if '_value_embeddings' not in key}
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
        self.setsumbt = self.model_0.setsumbt

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
                reduction: str = 'mean',
                **kwargs) -> tuple:
        """
        Args:
            input_ids: Input token ids
            attention_mask: Input padding mask
            token_type_ids: Token type indicator
            reduction: Reduction of ensemble member predictive distributions (mean, none)

        Returns:

        """
        belief_state_probs = {slot: [] for slot in self.setsumbt.config.informable_slot_ids}
        request_probs = {slot: [] for slot in self.setsumbt.config.requestable_slot_ids}
        active_domain_probs = {dom: [] for dom in self.setsumbt.config.domain_ids}
        general_act_probs = []
        loss = 0.0 if 'state_labels' in kwargs else None
        for attr in [f'model_{i}' for i in range(self.config.ensemble_size)]:
            # Prediction from each ensemble member
            with torch.no_grad():
                _out = getattr(self, attr)(input_ids=input_ids,
                                           token_type_ids=token_type_ids,
                                           attention_mask=attention_mask,
                                           **kwargs)
            if loss is not None:
                loss += _out.loss
            for slot in belief_state_probs:
                belief_state_probs[slot].append(_out.belief_state[slot].unsqueeze(-2).detach().cpu())
            if self.config.predict_actions:
                for slot in request_probs:
                    request_probs[slot].append(_out.request_probabilities[slot].unsqueeze(-1).detach().cpu())
                for dom in active_domain_probs:
                    active_domain_probs[dom].append(_out.active_domain_probabilities[dom].unsqueeze(-1).detach().cpu())
                general_act_probs.append(_out.general_act_probabilities.unsqueeze(-2).detach().cpu())

        belief_state_probs = {slot: torch.cat(l, -2) for slot, l in belief_state_probs.items()}
        if self.config.predict_actions:
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
            raise (NameError('Not Implemented!'))

        if loss is not None:
            loss /= self.config.ensemble_size

        output = SetSUMBTOutput(loss=loss,
                                belief_state=belief_state_probs,
                                request_probabilities=request_probs,
                                active_domain_probabilities=active_domain_probs,
                                general_act_probabilities=general_act_probs)

        return output

    @staticmethod
    def _get_checkpoint_path(path: str, idx: int):
        """
        Get checkpoint path for ensemble member
        Args:
            path: Location of ensemble
            idx: Ensemble member index

        Returns:
            Checkpoint path
        """

        checkpoints = os.listdir(os.path.join(path, f'ens-{idx}'))
        checkpoints = [int(p.split('-', 1)[-1]) for p in checkpoints if 'checkpoint-' in p]
        checkpoint = f"checkpoint-{max(checkpoints)}"
        return os.path.join(path, f'ens-{idx}', checkpoint)

    @classmethod
    def from_pretrained(cls, path, config=None):
        config_path = os.path.join(cls._get_checkpoint_path(path, 0), 'config.json')
        if not os.path.exists(config_path):
            raise (NameError('Could not find config.json in model path.'))

        if config is None:
            try:
                config = RobertaConfig.from_pretrained(config_path)
            except:
                config = BertConfig.from_pretrained(config_path)

        config.ensemble_size = len([dir for dir in os.listdir(path) if 'ens-' in dir])

        model = cls(config)
        model._load(path)

        return model
