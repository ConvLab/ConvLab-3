# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import TripletMarginLoss
from torch.nn import PairwiseDistance
from torch.nn import MultiheadAttention
import torch.nn.functional as F

#from transformers.file_utils import (add_start_docstrings, add_start_docstrings_to_callable)
#from transformers.modeling_utils import (PreTrainedModel)
#from transformers.modeling_roberta import (RobertaModel, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
#                                           ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING, BertLayerNorm)
from transformers import (RobertaModel, RobertaConfig, RobertaPreTrainedModel)

import time


#class RobertaPreTrainedModel(PreTrainedModel):
#    """ An abstract class to handle weights initialization and
#        a simple interface for dowloading and loading pretrained models.
#    """
#    config_class = RobertaConfig
#    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#    base_model_prefix = "roberta"
#    
#    def _init_weights(self, module):
#        """ Initialize the weights """
#        if isinstance(module, (nn.Linear, nn.Embedding)):
#            # Slightly different from the TF version which uses truncated_normal for initialization
#            # cf https://github.com/pytorch/pytorch/pull/5617
#            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#        elif isinstance(module, BertLayerNorm):
#            module.bias.data.zero_()
#            module.weight.data.fill_(1.0)
#        if isinstance(module, nn.Linear) and module.bias is not None:
#            module.bias.data.zero_()


#@add_start_docstrings(
#    """RoBERTa Model with classification heads for the DST task. """,
#    ROBERTA_START_DOCSTRING,
#)
class RobertaForDST(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.noncategorical = config.dst_noncategorical
        self.categorical = config.dst_categorical
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.value_loss_for_nonpointable = config.dst_value_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.stack_token_logits = False # config.dst_stack_token_logits # TODO
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio
        self.slot_attention_heads = config.dst_slot_attention_heads
        self.dropout_rate = config.dst_dropout_rate
        self.heads_dropout_rate = config.dst_heads_dropout_rate
        
        self.debug_fix_slot_embs = config.debug_fix_slot_embs
        self.debug_joint_slot_gate = config.debug_joint_slot_gate
        self.debug_joint_refer_gate = config.debug_joint_refer_gate
        self.debug_simple_joint_slot_gate = config.debug_simple_joint_slot_gate
        self.debug_separate_seq_tagging = config.debug_separate_seq_tagging
        self.debug_sigmoid_sequence_tagging = config.debug_sigmoid_sequence_tagging
        self.debug_att_output = config.debug_att_output
        self.debug_tanh_for_att_output = config.debug_tanh_for_att_output
        self.debug_stack_att = config.debug_stack_att
        self.debug_stack_rep = config.debug_stack_rep
        self.debug_sigmoid_slot_gates = config.debug_sigmoid_slot_gates
        self.debug_att_slot_gates = config.debug_att_slot_gates
        self.debug_use_triplet_loss = config.debug_use_triplet_loss
        self.debug_use_tlf = config.debug_use_tlf
        self.debug_value_att_none_class = config.debug_value_att_none_class
        self.debug_tag_none_target = config.debug_tag_none_target
        self.triplet_loss_weight = config.triplet_loss_weight
        self.none_weight = config.none_weight
        self.pretrain_loss_function = config.pretrain_loss_function
        self.token_loss_function = config.token_loss_function
        self.value_loss_function = config.value_loss_function
        self.class_loss_function = config.class_loss_function
        self.sequence_tagging_dropout = -1
        if config.sequence_tagging_dropout > 0.0:
            self.sequence_tagging_dropout = int(1 / config.sequence_tagging_dropout)
        self.ignore_o_tags = config.ignore_o_tags
        self.debug_slot_embs_per_step_nograd = config.debug_slot_embs_per_step_nograd
        try:
            self.debug_use_cross_attention = config.debug_use_cross_attention
        except:
            self.debug_use_cross_attention = False

        self.val_rep_mode = config.val_rep_mode # TODO: for debugging at the moment.
 
        config.output_hidden_states = True # TODO

        #config.dst_dropout_rate = 0.0 # TODO: for debugging
        #config.dst_heads_dropout_rate = 0.0 # TODO: for debugging
        
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        # -- Dialogue state tracking functionality --

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2

        # Slot specific gates
        for slot in self.slot_list:
            if not self.debug_joint_slot_gate:
                if self.debug_stack_att:
                    if self.debug_sigmoid_slot_gates:
                        for cl in range(self.class_labels):
                            self.add_module("class_" + slot + "_" + str(cl), nn.Linear(config.hidden_size * 2 + aux_dims, 1))
                    elif self.debug_att_slot_gates:
                        self.add_module("class_" + slot, MultiheadAttention(config.hidden_size * 2 + aux_dims, self.slot_attention_heads))
                    else:
                        self.add_module("class_" + slot, nn.Linear(config.hidden_size * 2 + aux_dims, self.class_labels))
                else:
                    if self.debug_sigmoid_slot_gates:
                        for cl in range(self.class_labels):
                            self.add_module("class_" + slot + "_" + str(cl), nn.Linear(config.hidden_size + aux_dims, 1))
                    elif self.debug_att_slot_gates:
                        self.add_module("class_" + slot, MultiheadAttention(config.hidden_size + aux_dims, self.slot_attention_heads))
                    else:
                        self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims, self.class_labels))
            #self.add_module("token_" + slot, nn.Linear(config.hidden_size, 1))
            if not self.debug_joint_refer_gate:
                self.add_module("refer_" + slot, nn.Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))
            if self.debug_separate_seq_tagging:
                if self.debug_sigmoid_sequence_tagging:
                    self.add_module("token_" + slot, nn.Linear(config.hidden_size, 1))
                else:
                    self.add_module("token_" + slot, MultiheadAttention(config.hidden_size, self.slot_attention_heads))

        if self.debug_att_output:
            self.class_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
        # Conditioned sequence tagging
        if not self.debug_separate_seq_tagging:
            if self.debug_sigmoid_sequence_tagging:
                self.h1t = nn.Linear(config.hidden_size, config.hidden_size)
                self.h2t = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
                self.llt = nn.Linear(config.hidden_size * 2, 1)
            else:
                self.token_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
        if self.debug_joint_refer_gate:
            self.refer_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
        self.value_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)
        #self.slot_att = MultiheadAttention(config.hidden_size, self.slot_attention_heads)

        self.token_layer_norm = nn.LayerNorm(config.hidden_size)
        self.token_layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.class_layer_norm = nn.LayerNorm(config.hidden_size)
        self.class_layer_norm2 = nn.LayerNorm(config.hidden_size)

        self.tlinear = nn.Linear(config.hidden_size, config.hidden_size)
        self.clinear = nn.Linear(config.hidden_size, config.hidden_size)

        # Conditioned slot gate
        self.h1c = nn.Linear(config.hidden_size + aux_dims, config.hidden_size)
        if self.debug_stack_att:
            self.h0c = nn.Linear(config.hidden_size, config.hidden_size)
            self.h2c = nn.Linear(config.hidden_size * 3, config.hidden_size * 3)
            if self.debug_sigmoid_slot_gates:
                for cl in range(self.class_labels):
                    self.add_module("llc_" + str(cl), nn.Linear(config.hidden_size * 3, 1))
            elif self.debug_att_slot_gates:
                if self.debug_att_output and self.debug_simple_joint_slot_gate:
                    self.llc = MultiheadAttention(config.hidden_size * 2, self.slot_attention_heads)
                else:
                    self.llc = MultiheadAttention(config.hidden_size * 3, self.slot_attention_heads)
            else:
                if self.debug_att_output and self.debug_simple_joint_slot_gate:
                    self.llc = nn.Linear(config.hidden_size * 2, self.class_labels)
                else:
                    self.llc = nn.Linear(config.hidden_size * 3, self.class_labels)
        else:
            if self.debug_att_slot_gates:
                self.h2c = nn.Linear(config.hidden_size * 2, config.hidden_size * 1)
            else:
                self.h2c = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
            if self.debug_sigmoid_slot_gates:
                for cl in range(self.class_labels):
                    self.add_module("llc_" + str(cl), nn.Linear(config.hidden_size * 2, 1))
            elif self.debug_att_slot_gates:
                if self.debug_att_output and self.debug_simple_joint_slot_gate:
                    self.llc = MultiheadAttention(config.hidden_size * 1, self.slot_attention_heads)
                else:
                    self.llc = MultiheadAttention(config.hidden_size * 2, self.slot_attention_heads)
            else:
                if self.debug_att_output and self.debug_simple_joint_slot_gate:
                    self.llc = nn.Linear(config.hidden_size * 1, self.class_labels)
                else:
                    self.llc = nn.Linear(config.hidden_size * 2, self.class_labels)

        # Conditioned refer gate
        self.h2r = nn.Linear(config.hidden_size * 2, config.hidden_size * 1)

        # -- Spanless sequence tagging functionality --

        self.dis = PairwiseDistance(p=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.binary_cross_entropy = F.binary_cross_entropy
        self.triplet_loss = TripletMarginLoss(margin=1.0, reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.refer_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=len(self.slot_list)) # Ignore 'none' target
        self.value_loss_fct = CrossEntropyLoss(reduction='none')

        if self.none_weight != 1.0:
            none_weight = self.none_weight
            weight_mass = none_weight + (self.class_labels - 1)
            none_weight /= weight_mass
            other_weights = 1 / weight_mass
            #self.clweights = torch.tensor([none_weight] + [other_weights] * (self.class_labels - 1)) # .to(outputs[0].device)
            self.clweights = torch.tensor([other_weights] * self.class_labels) # .to(outputs[0].device)
            self.clweights[self.class_types.index('none')] = none_weight
            if self.debug_sigmoid_slot_gates:
                self.class_loss_fct = F.binary_cross_entropy # (reduction="none") # weights for classes are controlled when adding losses
            else:
                self.class_loss_fct = CrossEntropyLoss(weight=self.clweights, reduction='none')
        else:
            if self.debug_sigmoid_slot_gates:
                self.class_loss_fct = F.binary_cross_entropy # (reduction="none")
            else:
                self.class_loss_fct = CrossEntropyLoss(reduction='none')

        self.init_weights()

    def compute_triplet_loss_att(self, att_output, pos_sampling_input, neg_sampling_input, slot):
        loss = self.triplet_loss(att_output, pos_sampling_input[slot].squeeze(), neg_sampling_input[slot].squeeze())
        sample_mask = neg_sampling_input[slot].squeeze(1).sum(1) != 0
        loss *= sample_mask
        return loss

    #@add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self, batch, step=None, epoch=None, t_slot=None, mode=None):
        assert(mode in [None, "pretrain", "tag", "encode", "encode_vals", "represent"]) # TODO

        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        #segment_ids = batch['segment_ids']
        usr_mask = batch['usr_mask']
        start_pos = batch['start_pos']
        end_pos = batch['end_pos']
        refer_id = batch['refer_id']
        class_label_id = batch['class_label_id']
        inform_slot_id = batch['inform_slot_id']
        diag_state = batch['diag_state']
        slot_ids = batch['slot_ids'] if 'slot_ids' in batch else None # TODO: fix?
        slot_mask = batch['slot_mask'] if 'slot_mask' in batch else None # TODO: fix?
        cl_ids = batch['cl_ids'] if 'cl_ids' in batch else None # TODO: fix?
        cl_mask = batch['cl_mask'] if 'cl_mask' in batch else None # TODO: fix?
        pos_sampling_input = batch['pos_sampling_input']
        neg_sampling_input = batch['neg_sampling_input']
        value_labels = batch['value_labels'] if 'value_labels' in batch else None # TODO: fix?

        batch_input_mask = input_mask
        if slot_ids is not None and slot_mask is not None:
            if self.debug_slot_embs_per_step_nograd:
                with torch.no_grad():
                    outputs_slot = self.roberta(
                        slot_ids,
                        attention_mask=slot_mask,
                        token_type_ids=None, # segment_ids,
                        position_ids=None,
                        head_mask=None
                    )
            else:
                input_ids = torch.cat((input_ids, slot_ids))
                input_mask = torch.cat((input_mask, slot_mask))
        if cl_ids is not None and cl_mask is not None:
            input_ids = torch.cat((input_ids, cl_ids))
            input_mask = torch.cat((input_mask, cl_mask))

        outputs = self.roberta(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=None, # segment_ids,
            position_ids=None,
            head_mask=None
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if cl_ids is not None and cl_mask is not None:
            sequence_output = sequence_output[:-1 * len(cl_ids), :, :]
            encoded_classes = pooled_output[-1 * len(cl_ids):, :]
            pooled_output = pooled_output[:-1 * len(cl_ids), :]
        if slot_ids is not None and slot_mask is not None:
            if self.debug_slot_embs_per_step_nograd:
                encoded_slots_seq = outputs_slot[0]
                encoded_slots_pooled = outputs_slot[1]
            else:
                encoded_slots_seq = sequence_output[-1 * len(slot_ids):, :, :]
                sequence_output = sequence_output[:-1 * len(slot_ids), :, :]
                encoded_slots_pooled = pooled_output[-1 * len(slot_ids):, :]
                pooled_output = pooled_output[:-1 * len(slot_ids), :]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        #sequence_output11 = outputs[2][-2]

        inverted_input_mask = ~(batch_input_mask.bool())
        if usr_mask is None:
            usr_mask = input_mask
        inverted_usr_mask = ~(usr_mask.bool())

        if mode == "encode": # Create vector representations only
            return pooled_output, sequence_output, None

        if mode == "encode_vals": # Create vector representations only
            no_seq_w = 1 / batch_input_mask.sum(1)
            uniform_weights = batch_input_mask * no_seq_w.unsqueeze(1)
            pooled_output_vals = torch.matmul(uniform_weights, sequence_output).squeeze(1)
            pooled_output_vals = self.token_layer_norm(pooled_output_vals)
            return pooled_output_vals, None, None
 
        if mode == "pretrain":
            pos_vectors = {}
            pos_weights = {}
            pos_vectors, pos_weights = self.token_att(
                query=encoded_slots_pooled.squeeze(1).unsqueeze(0),
                key=sequence_output.transpose(0, 1),
                value=sequence_output.transpose(0, 1),
                key_padding_mask=inverted_input_mask,
                need_weights=True)
            pos_vectors = pos_vectors.squeeze(0)
            pos_vectors = self.token_layer_norm(pos_vectors)
            pos_weights = pos_weights.squeeze(1)

            #neg_vectors = {}
            neg_weights = {}
            #neg_vectors[slot], neg_weights[slot] = self.token_att(
            #    query=neg_sampling_input[slot].squeeze(1).unsqueeze(0),
            #    key=sequence_output.transpose(0, 1),
            #    value=sequence_output.transpose(0, 1),
            #    key_padding_mask=inverted_input_mask,
            #    need_weights=True)
            #neg_vectors[slot] = neg_vectors[slot].squeeze(0)
            #neg_vectors[slot] = self.token_layer_norm(neg_vectors[slot])
            #neg_weights[slot] = neg_weights[slot].squeeze(1)

            pos_labels_clipped = torch.clamp(start_pos.float(), min=0, max=1)
            pos_labels_clipped_scaled = pos_labels_clipped / torch.clamp(pos_labels_clipped.sum(1).unsqueeze(1), min=1) # scaled
            #no_seq_w = 1 / batch_input_mask.sum(1)
            #neg_labels_clipped = batch_input_mask * no_seq_w.unsqueeze(1)
            if self.pretrain_loss_function == "mse":
                pos_token_loss = self.mse(pos_weights, pos_labels_clipped_scaled) # TODO: MSE might be better for scaled targets
            else:
                pos_token_loss = self.binary_cross_entropy(pos_weights, pos_labels_clipped_scaled, reduction="none")
            if self.ignore_o_tags:
                pos_token_loss *= pos_labels_clipped # TODO: good idea?
            #mm = torch.clamp(start_pos[slot].float(), min=0, max=1)
            #mm = torch.clamp(mm * 10, min=1)
            #pos_token_loss *= mm
            pos_token_loss = pos_token_loss.sum(1)
            #neg_token_loss = self.mse(neg_weights[slot], neg_labels_clipped)
            #neg_token_loss = neg_token_loss.sum(1)

            #triplet_loss = self.compute_triplet_loss_att(pos_vectors[slot], pos_sampling_input, neg_sampling_input, slot)

            per_example_loss = pos_token_loss # + triplet_loss # + neg_token_loss
            total_loss = per_example_loss.sum()
            
            return (total_loss, pos_weights, neg_weights,)

        if mode == "tag":
            query = torch.stack(list(batch['value_reps'].values())).transpose(1, 2).reshape(-1, pooled_output.size()[0], pooled_output.size()[1])
            _, weights = self.token_att(query=query,
                                        key=sequence_output.transpose(0, 1),
                                        value=sequence_output.transpose(0, 1),
                                        key_padding_mask=inverted_input_mask + inverted_usr_mask,
                                        need_weights=True)
            return (weights,)

        aaa_time = time.time()
        # Attention for sequence tagging
        vectors = {}
        weights = {}
        for s_itr, slot in enumerate(self.slot_list):
            if slot_ids is not None and slot_mask is not None:
                encoded_slot_seq = encoded_slots_seq[s_itr]
                encoded_slot_pooled = encoded_slots_pooled[s_itr]
            else:
                encoded_slot_seq = batch['encoded_slots_seq'][slot]
                encoded_slot_pooled = batch['encoded_slots_pooled'][slot]
            if self.debug_separate_seq_tagging:
                if self.debug_sigmoid_sequence_tagging:
                    weights[slot] = self.sigmoid(getattr(self, "token_" + slot)(sequence_output)).squeeze(2)
                    vectors[slot] = torch.matmul(weights[slot], sequence_output)
                    vectors[slot] = torch.diagonal(vectors[slot]).transpose(0, 1)
                    vectors[slot] = self.token_layer_norm(vectors[slot])
                else:
                    if self.debug_use_cross_attention:
                        query = encoded_slot_seq.squeeze().unsqueeze(0).expand(sequence_output.size()).transpose(0, 1)
                    else:
                        query = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(0)
                    vectors[slot], weights[slot] = getattr(self, "token_" + slot)(
                        query=query,
                        key=sequence_output.transpose(0, 1),
                        value=sequence_output.transpose(0, 1),
                        key_padding_mask=inverted_input_mask + inverted_usr_mask,
                        need_weights=True) # TODO: use usr_mask better or worse?
                    vectors[slot] = vectors[slot].squeeze(0)
                    #vectors[slot] = vectors[slot] / torch.norm(vectors[slot], dim=1, keepdim=True) # Normalize?
                    if self.debug_tanh_for_att_output:
                        #vectors[slot] = self.tanh(vectors[slot])
                        vectors[slot] = self.token_layer_norm2(vectors[slot])
                        vectors[slot] = self.tanh(self.tlinear(vectors[slot]))
                    #vectors[slot] += pooled_output # Residual
                    vectors[slot] = self.token_layer_norm(vectors[slot])
                    #vectors[slot] = self.dropout_heads(vectors[slot])
                    weights[slot] = weights[slot].squeeze(1)
                    if self.debug_stack_rep:
                        vectors[slot] = torch.cat((pooled_output, vectors[slot]), 1)
            else:
                if self.debug_sigmoid_sequence_tagging:
                    # Conditioned sequence tagging
                    sequence_output_add = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(1).expand(sequence_output.size())
                    xxt = self.gelu(self.h1t(sequence_output))
                    yyt = self.gelu(self.h2t(torch.cat((sequence_output_add, xxt), 2)))
                    weights[slot] = self.sigmoid(self.llt(yyt)).squeeze(2)
                    vectors[slot] = torch.matmul(weights[slot], sequence_output)
                    vectors[slot] = torch.diagonal(vectors[slot]).transpose(0, 1)
                    vectors[slot] = self.token_layer_norm(vectors[slot])
                else:
                    if self.debug_use_cross_attention:
                        query = encoded_slot_seq.squeeze().unsqueeze(0).expand(sequence_output.size()).transpose(0, 1)
                    else:
                        query = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(0)
                    vectors[slot], weights[slot] = self.token_att(
                        query=query,
                        key=sequence_output.transpose(0, 1),
                        value=sequence_output.transpose(0, 1),
                        key_padding_mask=inverted_input_mask + inverted_usr_mask,
                        need_weights=True) # TODO: use usr_mask better or worse?
                    vectors[slot] = vectors[slot].squeeze(0)
                    if self.debug_use_cross_attention:
                        vectors[slot] = torch.mean(vectors[slot] * (batch_input_mask + usr_mask).transpose(0, 1).unsqueeze(-1), dim=0)
                    #vectors[slot] = vectors[slot] / torch.norm(vectors[slot], dim=1, keepdim=True) # Normalize?
                    if self.debug_tanh_for_att_output:
                        #vectors[slot] = self.tanh(vectors[slot])
                        vectors[slot] = self.token_layer_norm2(vectors[slot])
                        vectors[slot] = self.tanh(self.tlinear(vectors[slot]))
                    #vectors[slot] += pooled_output # Residual
                    vectors[slot] = self.token_layer_norm(vectors[slot])
                    #vectors[slot] = self.dropout_heads(vectors[slot])
                    if self.debug_use_cross_attention:
                        weights[slot] = torch.mean(weights[slot] * (batch_input_mask + usr_mask).unsqueeze(-1), dim=1)
                    weights[slot] = weights[slot].squeeze(1)
                    if self.debug_stack_rep:
                        vectors[slot] = torch.cat((pooled_output, vectors[slot]), 1)

        if mode == "represent": # Create vector representations only
            return vectors, None, weights

        # TODO: establish proper format in labels already?
        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)

        bbb_time = time.time()
        total_loss = 0
        total_cl_loss = 0
        total_tk_loss = 0
        total_tp_loss = 0
        per_slot_per_example_loss = {}
        per_slot_per_example_cl_loss = {}
        per_slot_per_example_tk_loss = {}
        per_slot_per_example_tp_loss = {}
        per_slot_att_weights = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_value_logits = {}
        per_slot_refer_logits = {}
        for s_itr, slot in enumerate(self.slot_list):
            #if t_slot is not None and slot != t_slot:
            #    continue
            if slot_ids is not None and slot_mask is not None:
                encoded_slot_seq = encoded_slots_seq[s_itr]
                encoded_slot_pooled = encoded_slots_pooled[s_itr]
            else:
                encoded_slot_seq = batch['encoded_slots_seq'][slot]
                encoded_slot_pooled = batch['encoded_slots_pooled'][slot]

            # Attention for slot gates
            if self.debug_att_output:
                if self.debug_use_cross_attention:
                    query = encoded_slot_seq.squeeze().unsqueeze(0).expand(sequence_output.size()).transpose(0, 1)
                else:
                    query = encoded_slot_pooled.expand(pooled_output.size()).unsqueeze(0)
                att_output, c_weights = self.class_att(
                    query=query,
                    key=sequence_output.transpose(0, 1),
                    value=sequence_output.transpose(0, 1),
                    key_padding_mask=inverted_input_mask,
                    need_weights=True)
                if self.debug_use_cross_attention:
                    att_output = torch.mean(att_output, dim=0)
                    c_weights = torch.mean(c_weights, dim=1)
                if self.debug_tanh_for_att_output:
                    #att_output = self.tanh(att_output)
                    att_output = self.class_layer_norm2(att_output)
                    att_output = self.tanh(self.clinear(att_output))
                att_output = self.class_layer_norm(att_output)
                att_output = self.dropout_heads(att_output)
                per_slot_att_weights[slot] = c_weights.squeeze(1)
            else:
                per_slot_att_weights[slot] = None

            # Conditioned slot gate, or separate slot gates
            if self.debug_joint_slot_gate:
                if self.debug_att_output:
                    if self.class_aux_feats_inform and self.class_aux_feats_ds:
                        xx = self.gelu(self.h1c(torch.cat((att_output.squeeze(0), self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)))
                    elif self.class_aux_feats_inform:
                        xx = self.gelu(self.h1c(torch.cat((att_output.squeeze(0), self.inform_projection(inform_labels)), 1)))
                    elif self.class_aux_feats_ds:
                        xx = self.gelu(self.h1c(torch.cat((att_output.squeeze(0), self.ds_projection(diag_state_labels)), 1)))
                    else:
                        xx = self.gelu(self.h1c(att_output.squeeze(0)))
                    if self.debug_stack_att:
                        x0 = self.gelu(self.h0c(pooled_output))
                else:
                    if self.class_aux_feats_inform and self.class_aux_feats_ds:
                        xx = self.gelu(self.h1c(torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)))
                    elif self.class_aux_feats_inform:
                        xx = self.gelu(self.h1c(torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)))
                    elif self.class_aux_feats_ds:
                        xx = self.gelu(self.h1c(torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)))
                    else:
                        xx = self.gelu(self.h1c(pooled_output))
                if self.debug_att_output and not self.debug_simple_joint_slot_gate and self.debug_stack_att:
                    yy = self.gelu(self.h2c(torch.cat((encoded_slot_pooled.expand(pooled_output.size()), x0, xx), 1)))
                elif self.debug_att_output and self.debug_simple_joint_slot_gate and self.debug_stack_att:
                    yy = torch.cat((pooled_output, att_output.squeeze(0)), 1)
                elif self.debug_att_output and self.debug_simple_joint_slot_gate:
                    yy = att_output.squeeze(0)
                else:
                    yy = self.gelu(self.h2c(torch.cat((encoded_slot_pooled.expand(pooled_output.size()), xx), 1)))
                slot_gate_input = yy
                slot_gate_layer = "llc"
            else:
                if self.debug_att_output:
                    if self.debug_stack_att:
                        if self.class_aux_feats_inform and self.class_aux_feats_ds:
                            slot_gate_input = torch.cat((pooled_output, att_output.squeeze(0), self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
                        elif self.class_aux_feats_inform:
                            slot_gate_input = torch.cat((pooled_output, att_output.squeeze(0), self.inform_projection(inform_labels)), 1)
                        elif self.class_aux_feats_ds:
                            slot_gate_input = torch.cat((pooled_output, att_output.squeeze(0), self.ds_projection(diag_state_labels)), 1)
                        else:
                            slot_gate_input = torch.cat((pooled_output, att_output.squeeze(0)), 1)
                    else:
                        if self.class_aux_feats_inform and self.class_aux_feats_ds:
                            slot_gate_input = torch.cat((att_output.squeeze(0), self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
                        elif self.class_aux_feats_inform:
                            slot_gate_input = torch.cat((att_output.squeeze(0), self.inform_projection(inform_labels),), 1)
                        elif self.class_aux_feats_ds:
                            slot_gate_input = torch.cat((att_output.squeeze(0), self.ds_projection(diag_state_labels)), 1)
                        else:
                            slot_gate_input = att_output.squeeze(0)
                else:
                    if self.class_aux_feats_inform and self.class_aux_feats_ds:
                        slot_gate_input = torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
                    elif self.class_aux_feats_inform:
                        slot_gate_input = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
                    elif self.class_aux_feats_ds:
                        slot_gate_input = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
                    else:
                        slot_gate_input = pooled_output
                slot_gate_layer = "class_" + slot

            # Conditioned refer gate, or separate refer gates
            if self.debug_joint_slot_gate and self.debug_joint_refer_gate:
                slot_refer_input = self.gelu(self.h2r(torch.cat((encoded_slot_pooled.expand(pooled_output.size()), xx), 1)))
            else:
                if self.class_aux_feats_inform and self.class_aux_feats_ds:
                    slot_refer_input = torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
                elif self.class_aux_feats_inform:
                    slot_refer_input = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
                elif self.class_aux_feats_ds:
                    slot_refer_input = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
                else:
                    slot_refer_input = pooled_output

            # Slot gate classification
            if self.debug_sigmoid_slot_gates:
                class_logits = []
                for cl in range(self.class_labels):
                    class_logits.append(self.sigmoid(getattr(self, slot_gate_layer + "_" + str(cl))(slot_gate_input)))
                class_logits = torch.stack(class_logits, 1).squeeze(-1)
            elif self.debug_att_slot_gates:
                # TODO: implement separate gates as well
                if cl_ids is not None and cl_mask is not None:
                    bla = encoded_classes.unsqueeze(1).expand(-1, pooled_output.size()[0], -1)
                else:
                    bla = torch.stack(list(batch['encoded_classes'].values())).expand(-1, pooled_output.size()[0], -1) # TODO
                #_, class_logits = self.slot_att(
                _, class_logits = getattr(self, slot_gate_layer)(
                    query=slot_gate_input.unsqueeze(0),
                    key=bla,
                    value=bla,
                    need_weights=True)
                class_logits = class_logits.squeeze(1)
            else:
                class_logits = getattr(self, slot_gate_layer)(slot_gate_input)

            class_logits = self.dropout_heads(class_logits)

            #token_logits = self.dropout_heads(getattr(self, 'token_' + slot)(sequence_output).squeeze(-1))
            token_weights = weights[slot]

            # ---

            if self.triplet_loss_weight > 0.0 and not self.debug_use_triplet_loss:
                slot_values = torch.stack(list(batch['encoded_slot_values'][slot].values())) # TODO: filter?
                slot_values = slot_values.expand((-1, pooled_output.size(0), -1))
                #yy = (batch['pos_sampling_input'][slot].sum(2) > 0.0)
                #if self.debug_value_att_none_class:
                #    ww = batch['value_labels'][slot][:, 1:] * yy
                #else:
                #    ww = batch['value_labels'][slot] * yy
                #wwx = ww == 0
                #bbb = slot_values * wwx.transpose(0, 1).unsqueeze(2)
                #wwy = ww == 1
                #yyy = batch['pos_sampling_input'][slot].expand(-1, slot_values.size(0), -1) * wwy.unsqueeze(2)
                #slot_values = bbb + yyy.transpose(0, 1)
                if self.debug_value_att_none_class:
                    slot_values = torch.cat((vectors[slot].unsqueeze(0), slot_values))
                #slot_values = slot_values.expand(-1, pooled_output.size()[0], -1)
                #slot_values = torch.stack((batch['pos_sampling_input'][slot], batch['neg_sampling_input'][slot])) # TODO: filter?
                #slot_values = torch.stack((vectors[slot].unsqueeze(1), batch['pos_sampling_input'][slot], batch['neg_sampling_input'][slot])) # TODO: filter?
                #slot_values = slot_values.squeeze(2)
                _, value_weights = self.value_att(
                    query=vectors[slot].unsqueeze(0),
                    key=slot_values,
                    value=slot_values,
                    need_weights=True)
                ##vectors[slot] = vectors[slot].squeeze(0)
                #vectors[slot] = torch.matmul(weights[slot], sequence_output).squeeze(1)
                #vectors[slot] = vectors[slot] / torch.norm(vectors[slot], dim=1, keepdim=True) # Normalize?
                #if self.debug_tanh_for_att_output:
                    #vectors[slot] = self.tanh(vectors[slot])
                    #vectors[slot] = self.token_layer_norm2(vectors[slot])
                    #vectors[slot] = self.tanh(self.tlinear(vectors[slot]))
                #vectors[slot] += pooled_output # Residual
                #vectors[slot] = self.token_layer_norm(vectors[slot])
                #vectors[slot] = self.dropout_heads(vectors[slot])
                value_weights = value_weights.squeeze(1)
            else:
                value_weights = None

            # ---

            # TODO: implement choice between joint_refer_gate and individual refer gates, analogous to
            # slot gates. Use same input as slot gate, i.e., for joint case, use yy, for individual
            # use pooled. This is stored in slot_gate_input.
            
            # ---

            if self.debug_joint_refer_gate:
                if slot_ids is not None and slot_mask is not None:
                    refer_slots = encoded_slots_pooled.unsqueeze(1).expand(-1, pooled_output.size()[0], -1)
                else:
                    #refer_slots = torch.stack((list(self.encoded_slots.values()))).expand(-1, pooled_output.size()[0], -1)
                    refer_slots = torch.stack(list(batch['encoded_slots_pooled'].values())).expand(-1, pooled_output.size()[0], -1)
                _, refer_weights = self.refer_att(
                    query=slot_refer_input.unsqueeze(0),
                    key=refer_slots,
                    value=refer_slots,
                    need_weights=True)
                refer_weights = refer_weights.squeeze(1)
                refer_logits = refer_weights
            else:
                refer_logits = getattr(self, "refer_" + slot)(slot_refer_input)

            refer_logits = self.dropout_heads(refer_logits)

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = token_weights # TODO
            per_slot_value_logits[slot] = value_weights
            #per_slot_start_logits[slot] = self.sigmoid(token_logits) # TODO # this is for sigmoid approach
            per_slot_refer_logits[slot] = refer_logits

            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)

                # TODO: solve this using the sequence_tagging def?
                labels_clipped = torch.clamp(start_pos[slot].float(), min=0, max=1)
                labels_clipped_scaled = labels_clipped / torch.clamp(labels_clipped.sum(1).unsqueeze(1), min=1) # Scale targets?
                no_seq_mask = labels_clipped_scaled.sum(1) == 0
                no_seq_w = 1 / batch_input_mask.sum(1)
                labels_clipped_scaled += batch_input_mask * (no_seq_mask * no_seq_w).unsqueeze(1)
                #token_weights = self.sigmoid(token_logits) # TODO # this is for sigmoid approach
                if self.token_loss_function == "mse":
                    token_loss = self.mse(token_weights, labels_clipped_scaled) # TODO: MSE might be better for scaled targets
                else:
                    token_loss = self.binary_cross_entropy(token_weights, labels_clipped_scaled, reduction="none")
                if self.ignore_o_tags:
                    token_loss *= labels_clipped # TODO: good idea?

                # TODO: do negative examples have to be balanced due to their large number?
                token_loss = token_loss.sum(1)
                token_is_pointable = (start_pos[slot].sum(1) > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                value_loss = torch.zeros(token_is_pointable.size(), device=token_is_pointable.device)
                if self.triplet_loss_weight > 0.0:
                    if self.debug_use_triplet_loss:
                        value_loss = self.compute_triplet_loss_att(vectors[slot], pos_sampling_input, neg_sampling_input, slot)
                        #triplet_loss = torch.clamp(triplet_loss, max=1) # TODO: parameterize # Not the best idea I think...
                    else:
                        value_labels_clipped = torch.clamp(value_labels[slot].float(), min=0, max=1)
                        value_labels_clipped /= torch.clamp(value_labels_clipped.sum(1).unsqueeze(1), min=1) # Scale targets?
                        value_no_seq_mask = value_labels_clipped.sum(1) == 0
                        value_no_seq_w = 1 / value_labels_clipped.size(1)
                        value_labels_clipped += (value_no_seq_mask * value_no_seq_w).unsqueeze(1)
                        if self.value_loss_function == "mse":
                            value_loss = self.mse(value_weights, value_labels_clipped) # TODO: scale value_labels to also cover nonpointable cases and multitarget cases
                        else:
                            value_loss = self.binary_cross_entropy(value_weights, value_labels_clipped, reduction="none")
                            #print(slot)
                            #print(value_labels_clipped)
                            #print(value_weights)
                            #print(value_loss)
                        value_loss = value_loss.sum(1)
                    token_is_matchable = token_is_pointable
                    if self.debug_tag_none_target:
                        token_is_matchable *= (start_pos[slot][:, 1] == 0).float()
                    if not self.value_loss_for_nonpointable:
                        value_loss *= token_is_matchable

                # Re-definition necessary to make slot-independent prediction possible
                self.refer_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=len(self.slot_list)) # Ignore 'none' target
                refer_loss = self.refer_loss_fct(refer_logits, refer_id[slot])
                token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                if not self.refer_loss_for_nonpointable:
                    refer_loss *= token_is_referrable

                if self.debug_sigmoid_slot_gates:
                    class_loss = []
                    for cl in range(self.class_labels):
                        #class_loss.append(self.binary_cross_entropy(class_logits[:,cl], labels, reduction="none"))
                        class_loss.append(self.class_loss_fct(class_logits[:, cl], (class_label_id[slot] == cl).float(), reduction="none"))
                    class_loss = torch.stack(class_loss, 1)
                    if self.none_weight != 1.0:
                        class_loss *= self.clweights.to(outputs[0].device)
                    class_loss = class_loss.sum(1)
                elif self.debug_att_slot_gates:
                    if self.class_loss_function == "mse":
                        class_loss = self.mse(class_logits, torch.nn.functional.one_hot(class_label_id[slot], self.class_labels).float())
                    else:
                        class_loss = self.binary_cross_entropy(class_logits, torch.nn.functional.one_hot(class_label_id[slot], self.class_labels).float(), reduction="none")
                    if self.none_weight != 1.0:
                        class_loss *= self.clweights.to(outputs[0].device)
                    class_loss = class_loss.sum(1)
                else:
                    class_loss = self.class_loss_fct(class_logits, class_label_id[slot])

                #print("%15s, class loss: %.3f, token loss: %.3f, triplet loss: %.3f" % (slot, (class_loss.sum()).item(), (token_loss.sum()).item(), (triplet_loss.sum()).item()))
                #print("%15s, class loss: %.3f, token loss: %.3f, value loss: %.3f" % (slot, (class_loss.sum()).item(), (token_loss.sum()).item(), (value_loss.sum()).item()))

                st_switch = int(not (self.sequence_tagging_dropout >= 1 and step is not None and step % self.sequence_tagging_dropout == 0))

                if self.refer_index > -1:
                    #per_example_loss = (self.class_loss_ratio) * class_loss + st_switch * ((1 - self.class_loss_ratio) / 2) * token_loss + ((1 - self.class_loss_ratio) / 2) * refer_loss + self.triplet_loss_weight * triplet_loss
                    per_example_loss = (self.class_loss_ratio) * class_loss + st_switch * ((1 - self.class_loss_ratio) / 2) * token_loss + ((1 - self.class_loss_ratio) / 2) * refer_loss + self.triplet_loss_weight * value_loss
                    #per_example_loss = class_loss
                else:
                    #per_example_loss = self.class_loss_ratio * class_loss + st_switch * (1 - self.class_loss_ratio) * token_loss + self.triplet_loss_weight * triplet_loss
                    per_example_loss = self.class_loss_ratio * class_loss + st_switch * (1 - self.class_loss_ratio) * token_loss + self.triplet_loss_weight * value_loss
                    #if epoch is not None and epoch > 20:
                    #    per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss + self.triplet_loss_weight * triplet_loss
                    #else:
                    #    per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss
                    #per_example_loss = class_loss
                if self.debug_use_tlf:
                    per_example_loss *= 1.0 + ((batch['diag_len'] - batch['turn_id']) * 0.05)

                total_loss += per_example_loss.sum()
                total_cl_loss += class_loss.sum()
                total_tk_loss += token_loss.sum()
                #total_tp_loss += triplet_loss.sum()
                total_tp_loss += value_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss
                per_slot_per_example_cl_loss[slot] = class_loss
                per_slot_per_example_tk_loss[slot] = token_loss
                #per_slot_per_example_tp_loss[slot] = triplet_loss
                per_slot_per_example_tp_loss[slot] = value_loss
        ccc_time = time.time()
        #print(bbb_time - aaa_time, ccc_time - bbb_time) # 0.028620243072509766

        # add hidden states and attention if they are here
        outputs = (total_loss,
                   total_cl_loss,
                   total_tk_loss,
                   total_tp_loss,
                   per_slot_per_example_loss,
                   per_slot_per_example_cl_loss,
                   per_slot_per_example_tk_loss,
                   per_slot_per_example_tp_loss,
                   per_slot_class_logits,
                   per_slot_start_logits,
                   per_slot_end_logits,
                   per_slot_value_logits,
                   per_slot_refer_logits,
                   per_slot_att_weights,) + (vectors, weights,) + (pooled_output,) # + outputs[2:]
        #outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits, per_slot_att_weights,) + (vectors, weights,) # + outputs[2:]

        return outputs
