# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
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
"""RoBERTa SetSUMBT"""

import torch
from transformers import RobertaModel, RobertaPreTrainedModel

from convlab.dst.setsumbt.modeling.setsumbt import SetSUMBTHead


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
        
        if get_turn_pooled_representation:
            return self.setsumbt(turn_embeddings, roberta_output.pooler_output, attention_mask,
                                 batch_size, dialogue_size, hidden_state, state_labels,
                                 request_labels, active_domain_labels, general_act_labels,
                                 calculate_state_mutual_info) + (roberta_output.pooler_output,)
        return self.setsumbt(turn_embeddings, roberta_output.pooler_output, attention_mask, batch_size,
                             dialogue_size, hidden_state, state_labels, request_labels, active_domain_labels,
                             general_act_labels, calculate_state_mutual_info)
