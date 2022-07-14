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
"""BERT SetSUMBT"""

import torch
import transformers
from torch.autograd import Variable
from transformers import BertModel, BertPreTrainedModel

from convlab.dst.setsumbt.modeling.functional import initialize_setsumbt_model, nbt_forward


class BertSetSUMBT(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSetSUMBT, self).__init__(config)
        self.config = config

        # Turn Encoder
        self.bert = BertModel(config)
        if config.freeze_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

        initialize_setsumbt_model(self, config)

    # Add new slot candidates to the model
    def add_slot_candidates(self, slot_candidates):
        """slot_candidates is a list of tuples for each slot.
        - The tuples contains the slot embedding, informable value embeddings and a request indicator.
        - If the informable value embeddings is None the slot is not informable
        - If the request indicator is false the slot is not requestable"""
        if self.slot_embeddings.size(0) != 0:
            embeddings = self.slot_embeddings.detach()
        else:
            embeddings = torch.zeros(0)

        for slot in slot_candidates:
            if slot in self.slot_ids:
                index = self.slot_ids[slot]
                embeddings[index, :] = slot_candidates[slot][0]
            else:
                index = embeddings.size(0)
                emb = slot_candidates[slot][0].unsqueeze(0).to(embeddings.device)
                embeddings = torch.cat((embeddings, emb), 0)
                self.slot_ids[slot] = index
                setattr(self, slot + '_value_embeddings', Variable(torch.zeros(0), requires_grad=False))
            # Add slot to relevant requestable and informable slot lists
            if slot_candidates[slot][2]:
                self.requestable_slot_ids[slot] = index
            if slot_candidates[slot][1] is not None:
                self.informable_slot_ids[slot] = index
            
            domain = slot.split('-', 1)[0]
            if domain not in self.domain_ids:
                self.domain_ids[domain] = []
            self.domain_ids[domain].append(index)
            self.domain_ids[domain] = list(set(self.domain_ids[domain]))
        
        self.slot_embeddings = Variable(embeddings, requires_grad=False)


    # Add new value candidates to the model
    def add_value_candidates(self, slot, value_candidates, replace=False):
        embeddings = getattr(self, slot + '_value_embeddings')

        if embeddings.size(0) == 0 or replace:
            embeddings = value_candidates
        else:
            embeddings = torch.cat((embeddings, value_candidates), 0)
        
        setattr(self, slot + '_value_embeddings', embeddings)

    
    def forward(self, input_ids, token_type_ids, attention_mask, hidden_state=None, inform_labels=None,
                request_labels=None, domain_labels=None, goodbye_labels=None,
                get_turn_pooled_representation=False, calculate_inform_mutual_info=False):

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

        if get_turn_pooled_representation:
            return nbt_forward(self, turn_embeddings, bert_output.pooler_output, attention_mask, batch_size,
                                dialogue_size, hidden_state, inform_labels, request_labels, domain_labels,
                                goodbye_labels, calculate_inform_mutual_info) + (bert_output.pooler_output,)
        return nbt_forward(self, turn_embeddings, bert_output.pooler_output, attention_mask, batch_size, dialogue_size,
                            hidden_state, inform_labels, request_labels, domain_labels, goodbye_labels,
                            calculate_inform_mutual_info)
