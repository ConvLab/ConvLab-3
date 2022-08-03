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
"""SetSUMBT functionals"""

import torch
from torch.autograd import Variable
from torch.nn import (MultiheadAttention, GRU, LSTM, Linear, LayerNorm, Dropout,
                      CosineSimilarity, CrossEntropyLoss, PairwiseDistance,
                      Sequential, ReLU, Conv1d, GELU, BCEWithLogitsLoss)
from torch.nn.init import (xavier_normal_, constant_)

from convlab.dst.setsumbt.loss import (BayesianMatchingLoss, BinaryBayesianMatchingLoss,
                                       KLDistillationLoss, BinaryKLDistillationLoss,
                                       LabelSmoothingLoss, BinaryLabelSmoothingLoss,
                                       RKLDirichletMediatorLoss, BinaryRKLDirichletMediatorLoss)


# Default belief tracker model initialization function
def initialize_setsumbt_model(self, config):
    """
    Initialisation of the SetSUMBT head

    Args:
        self (class instance): Model class instance
        config (configuration): Model configuration class
    """
    # Slot Utterance matching attention
    self.slot_attention = MultiheadAttention(config.hidden_size, config.slot_attention_heads)

    # Latent context tracker
    # Initial state prediction
    if not config.rnn_zero_init and config.nbt_type in ['gru', 'lstm']:
        self.belief_init = Sequential(Linear(config.hidden_size, config.nbt_hidden_size), ReLU(),
                                      Dropout(config.dropout_rate))

    # Recurrent context tracker setup
    if config.nbt_type == 'gru':
        self.nbt = GRU(input_size=config.hidden_size,
                       hidden_size=config.nbt_hidden_size,
                       num_layers=config.nbt_layers,
                       dropout=0.0 if config.nbt_layers == 1 else config.dropout_rate,
                       batch_first=True)
        # Initialise Parameters
        xavier_normal_(self.nbt.weight_ih_l0)
        xavier_normal_(self.nbt.weight_hh_l0)
        constant_(self.nbt.bias_ih_l0, 0.0)
        constant_(self.nbt.bias_hh_l0, 0.0)
    elif config.nbt_type == 'lstm':
        self.nbt = LSTM(input_size=config.hidden_size,
                        hidden_size=config.nbt_hidden_size,
                        num_layers=config.nbt_layers,
                        dropout=0.0 if config.nbt_layers == 1 else config.dropout_rate,
                        batch_first=True)
        # Initialise Parameters
        xavier_normal_(self.nbt.weight_ih_l0)
        xavier_normal_(self.nbt.weight_hh_l0)
        constant_(self.nbt.bias_ih_l0, 0.0)
        constant_(self.nbt.bias_hh_l0, 0.0)
    else:
        raise NameError('Not Implemented')

    # Intermediate feature mapping and layer normalisation
    self.intermediate = Linear(config.nbt_hidden_size, config.hidden_size)
    self.layer_norm = LayerNorm(config.hidden_size)

    # Dropout
    self.dropout = Dropout(config.dropout_rate)

    # Set pooler for set similarity model
    if self.config.set_similarity:
        # 1D convolutional set pooler
        if self.config.set_pooling == 'cnn':
            self.conv_pooler = Conv1d(self.config.hidden_size, self.config.hidden_size, 3)
        # Deep averaging network set pooler
        elif self.config.set_pooling == 'dan':
            self.avg_net = Sequential(Linear(self.config.hidden_size, 2 * self.config.hidden_size), GELU(),
                                      Linear(2 * self.config.hidden_size, self.config.hidden_size))

    # Model ontology placeholders
    self.slot_embeddings = Variable(torch.zeros(0), requires_grad=False)
    self.slot_ids = dict()
    self.requestable_slot_ids = dict()
    self.informable_slot_ids = dict()
    self.domain_ids = dict()

    # Matching network similarity measure
    if config.distance_measure == 'cosine':
        self.distance = CosineSimilarity(dim=-1, eps=1e-8)
    elif config.distance_measure == 'euclidean':
        self.distance = PairwiseDistance(p=2.0, eps=1e-6, keepdim=False)
    else:
        raise NameError('NotImplemented')

    # User goal prediction loss function
    if config.loss_function == 'crossentropy':
        self.loss = CrossEntropyLoss(ignore_index=-1)
    elif config.loss_function == 'bayesianmatching':
        self.loss = BayesianMatchingLoss(ignore_index=-1, lamb=config.kl_scaling_factor)
    elif config.loss_function == 'labelsmoothing':
        self.loss = LabelSmoothingLoss(ignore_index=-1, label_smoothing=config.label_smoothing)
    elif config.loss_function == 'distillation':
        self.loss = KLDistillationLoss(ignore_index=-1, lamb=config.ensemble_smoothing)
        self.temp = 1.0
    elif config.loss_function == 'distribution_distillation':
        self.loss = RKLDirichletMediatorLoss(ignore_index=-1)
    else:
        raise NameError('NotImplemented')

    # Intent and domain prediction heads
    if config.predict_actions:
        self.request_gate = Linear(config.hidden_size, 1)
        self.general_act_gate = Linear(config.hidden_size, 3)
        self.active_domain_gate = Linear(config.hidden_size, 1)

        # Intent and domain loss function
        self.request_weight = float(self.config.user_request_loss_weight)
        self.general_act_weight = float(self.config.user_general_act_loss_weight)
        self.active_domain_weight = float(self.config.active_domain_loss_weight)
        if config.loss_function == 'crossentropy':
            self.request_loss = BCEWithLogitsLoss()
            self.general_act_loss = CrossEntropyLoss(ignore_index=-1)
            self.active_domain_loss = BCEWithLogitsLoss()
        elif config.loss_function == 'labelsmoothing':
            self.request_loss = BinaryLabelSmoothingLoss(label_smoothing=config.label_smoothing)
            self.general_act_loss = LabelSmoothingLoss(ignore_index=-1, label_smoothing=config.label_smoothing)
            self.active_domain_loss = BinaryLabelSmoothingLoss(label_smoothing=config.label_smoothing)
        elif config.loss_function == 'bayesianmatching':
            self.request_loss = BinaryBayesianMatchingLoss(ignore_index=-1, lamb=config.kl_scaling_factor)
            self.general_act_loss = BayesianMatchingLoss(ignore_index=-1, lamb=config.kl_scaling_factor)
            self.active_domain_loss = BinaryBayesianMatchingLoss(ignore_index=-1, lamb=config.kl_scaling_factor)
        elif config.loss_function == 'distillation':
            self.request_loss = BinaryKLDistillationLoss(ignore_index=-1, lamb=config.ensemble_smoothing)
            self.general_act_loss = KLDistillationLoss(ignore_index=-1, lamb=config.ensemble_smoothing)
            self.active_domain_loss = BinaryKLDistillationLoss(ignore_index=-1, lamb=config.ensemble_smoothing)
        elif config.loss_function == 'distribution_distillation':
            self.request_loss = BinaryRKLDirichletMediatorLoss(ignore_index=-1)
            self.general_act_loss = RKLDirichletMediatorLoss(ignore_index=-1)
            self.active_domain_loss = BinaryRKLDirichletMediatorLoss(ignore_index=-1)


# Default belief tracker forward pass.
def nbt_forward(self,
                turn_embeddings: torch.Tensor,
                turn_pooled_representation: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_size: int,
                dialogue_size: int,
                hidden_state: torch.Tensor = None,
                state_labels: torch.Tensor = None,
                request_labels: torch.Tensor = None,
                active_domain_labels: torch.Tensor = None,
                general_act_labels: torch.Tensor = None,
                calculate_inform_mutual_info: bool = False):
    hidden_size = turn_embeddings.size(-1)
    # Initialise loss
    loss = 0.0

    # General Action predictions
    goodbye_probs = None
    if self.config.predict_actions:
        # General action prediction
        goodbye_scores = self.general_act_gate(turn_pooled_representation.reshape(batch_size * dialogue_size,
                                                                                  hidden_size))

        # Compute loss for general action predictions (weighted loss)
        if general_act_labels is not None:
            if self.config.loss_function == 'distillation':
                general_act_labels = general_act_labels.reshape(-1, general_act_labels.size(-1))
                loss += self.general_act_loss(goodbye_scores, general_act_labels, self.temp) * self.general_act_weight
            elif self.config.loss_function == 'distribution_distillation':
                general_act_labels = general_act_labels.reshape(-1, general_act_labels.size(-2),
                                                                general_act_labels.size(-1))
                loss += self.general_act_loss(goodbye_scores, general_act_labels, 1.0, 1.0)[0] * self.general_act_weight
            else:
                general_act_labels = general_act_labels.reshape(-1)
                loss += self.general_act_loss(goodbye_scores, general_act_labels) * self.request_weight

        # Compute general action probabilities
        if self.config.loss_function in ['crossentropy', 'labelsmoothing', 'distillation', 'distribution_distillation']:
            goodbye_probs = torch.softmax(goodbye_scores, -1).reshape(batch_size, dialogue_size, -1)
        elif self.config.loss_function in ['bayesianmatching']:
            goodbye_probs = torch.sigmoid(goodbye_scores.reshape(batch_size, dialogue_size, -1))

    # Slot utterance matching
    num_slots = self.slot_embeddings.size(0)
    slot_embeddings = self.slot_embeddings.reshape(-1, hidden_size)
    slot_embeddings = slot_embeddings.unsqueeze(1).repeat((1, batch_size * dialogue_size, 1)).to(turn_embeddings.device)

    if self.config.set_similarity:
        # Slot mask shape [num_slots * slot_len, batch_size * dialogue_size, 768]
        slot_mask = (slot_embeddings != 0.0).float()

    # Turn embeddings shape [turn_size, batch_size * dialogue_size, 768]
    turn_embeddings = turn_embeddings.transpose(0, 1)
    # Compute key padding mask
    key_padding_mask = (attention_mask[:, :, 0] == 0.0)
    key_padding_mask[key_padding_mask[:, 0], :] = False
    # Multi head attention of slot over tokens
    hidden, _ = self.slot_attention(query=slot_embeddings,
                                    key=turn_embeddings,
                                    value=turn_embeddings,
                                    key_padding_mask=key_padding_mask)  # [num_slots, batch_size * dialogue_size, 768]

    # Set embeddings for all masked tokens to 0
    attention_mask = attention_mask[:, 0, :].unsqueeze(0).repeat((slot_embeddings.size(0), 1, 1))
    hidden = hidden * attention_mask
    if self.config.set_similarity:
        hidden = hidden * slot_mask
    # Hidden layer shape [num_dials, num_slots, num_turns, 768]
    hidden = hidden.transpose(0, 1).reshape(batch_size, dialogue_size, slot_embeddings.size(0), -1).transpose(1, 2)

    # Latent context tracking
    # [batch_size * num_slots, dialogue_size, 768]
    hidden = hidden.reshape(batch_size * slot_embeddings.size(0), dialogue_size, -1)

    if self.config.nbt_type == 'gru':
        self.nbt.flatten_parameters()
        if hidden_state is None:
            if self.config.rnn_zero_init:
                context = torch.zeros(self.config.nbt_layers, batch_size * slot_embeddings.size(0),
                                      self.config.nbt_hidden_size)
                context = context.to(turn_embeddings.device)
            else:
                context = self.belief_init(hidden[:, 0, :]).unsqueeze(0).repeat((self.config.nbt_layers, 1, 1))
        else:
            context = hidden_state.to(hidden.device)

        # [batch_size, dialogue_size, nbt_hidden_size]
        belief_embedding, context = self.nbt(hidden, context)
    elif self.config.nbt_type == 'lstm':
        self.nbt.flatten_parameters()
        if self.config.rnn_zero_init:
            context = (torch.zeros(self.config.nbt_layers, batch_size * num_slots, self.config.nbt_hidden_size),
                       torch.zeros(self.config.nbt_layers, batch_size * num_slots, self.config.nbt_hidden_size))
            context = (context[0].to(turn_embeddings.device),
                       context[1].to(turn_embeddings.device))
        else:
            context = (self.belief_init(hidden[:, 0, :]).unsqueeze(0).repeat((self.config.nbt_layers, 1, 1)),
                       torch.zeros(self.config.nbt_layers, batch_size * num_slots, self.config.nbt_hidden_size))
            context = (context[0], context[1].to(turn_embeddings.device))

        # [batch_size, dialogue_size, nbt_hidden_size]
        belief_embedding, context = self.nbt(hidden, context)

    # Intermediate feature transformation
    belief_embedding = belief_embedding.reshape(batch_size, slot_embeddings.size(0), dialogue_size, -1).transpose(1, 2)
    if self.config.set_similarity:
        belief_embedding = belief_embedding.reshape(batch_size, dialogue_size, num_slots, -1,
                                                    self.config.nbt_hidden_size)
    # [batch_size, dialogue_size, num_slots, *slot_desc_len, 768]
    # Normalisation and regularisation
    belief_embedding = self.layer_norm(self.intermediate(belief_embedding))
    belief_embedding = self.dropout(belief_embedding)

    # Pooling of the set of latent context representation
    if self.config.set_similarity:
        slot_mask = slot_mask.transpose(0, 1).reshape(batch_size, dialogue_size, num_slots, -1, hidden_size)
        belief_embedding = belief_embedding * slot_mask

        # Apply pooler to latent context sequence
        if self.config.set_pooling == 'mean':
            belief_embedding = belief_embedding.sum(-2) / slot_mask.sum(-2)
            belief_embedding = belief_embedding.reshape(batch_size, dialogue_size, num_slots, -1)
        elif self.config.set_pooling == 'cnn':
            belief_embedding = belief_embedding.reshape(-1, slot_mask.size(-2), hidden_size).transpose(1, 2)
            belief_embedding = self.conv_pooler(belief_embedding)
            # Mean pooling after CNN
            belief_embedding = belief_embedding.mean(-1).reshape(batch_size, dialogue_size, num_slots, -1)
        elif self.config.set_pooling == 'dan':
            # sqrt N reduction
            belief_embedding = belief_embedding.sum(-2) / torch.sqrt(torch.tensor(slot_mask.sum(-2)))
            # Deep averaging feature extractor
            belief_embedding = self.avg_net(belief_embedding)
            belief_embedding = belief_embedding.reshape(batch_size, dialogue_size, num_slots, -1)

    # Perform classification
    if self.config.predict_actions:
        # User request prediction
        request_probs = dict()
        for slot, slot_id in self.requestable_slot_ids.items():
            request_scores = self.request_gate(belief_embedding[:, :, slot_id, :])

            # Store output probabilities
            request_scores = request_scores.reshape(batch_size, dialogue_size)
            mask = attention_mask[0, :, 0].reshape(batch_size, dialogue_size)
            batches, dialogues = torch.where(mask == 0.0)
            # Set request scores to 0.0 for padded turns
            request_scores[batches, dialogues] = 0.0
            if self.config.loss_function in ['crossentropy', 'labelsmoothing', 'bayesianmatching',
                                             'distillation', 'distribution_distillation']:
                request_probs[slot] = torch.sigmoid(request_scores)

            if request_labels is not None:
                # Compute request gate loss
                request_scores = request_scores.reshape(-1)
                if self.config.loss_function == 'distillation':
                    loss += self.request_loss(request_scores, request_labels[slot].reshape(-1),
                                              self.temp) * self.request_weight
                elif self.config.loss_function == 'distribution_distillation':
                    scores, labs = convert_probs_to_logits(request_scores, request_labels[slot])
                    loss += self.loss(scores, labs, 1.0, 1.0)[0] * self.request_weight
                else:
                    labs = request_labels[slot].reshape(-1)
                    request_scores = request_scores[labs != -1]
                    labs = labs[labs != -1].float()
                    loss += self.request_loss(request_scores, labs) * self.request_weight

        # Active domain prediction
        domain_probs = dict()
        for domain, slot_ids in self.domain_ids.items():
            belief = belief_embedding[:, :, slot_ids, :]
            if len(slot_ids) > 1:
                # SqrtN reduction across all slots within a domain
                belief = belief.sum(2) / ((belief != 0.0).float().sum(2) ** 0.5)
            domain_scores = self.active_domain_gate(belief)

            # Store output probabilities
            domain_scores = domain_scores.reshape(batch_size, dialogue_size)
            mask = attention_mask[0, :, 0].reshape(batch_size, dialogue_size)
            batches, dialogues = torch.where(mask == 0.0)
            domain_scores[batches, dialogues] = 0.0
            if self.config.loss_function in ['crossentropy', 'labelsmoothing', 'bayesianmatching', 'distillation',
                                             'distribution_distillation']:
                domain_probs[domain] = torch.sigmoid(domain_scores)

            if active_domain_labels is not None and domain in active_domain_labels:
                # Compute domain prediction loss
                domain_scores = domain_scores.reshape(-1)
                if self.config.loss_function == 'distillation':
                    loss += self.active_domain_loss(domain_scores, active_domain_labels[domain].reshape(-1),
                                                    self.temp) * self.active_domain_weight
                elif self.config.loss_function == 'distribution_distillation':
                    scores, labs = convert_probs_to_logits(domain_scores, active_domain_labels[domain])
                    loss += self.loss(scores, labs, 1.0, 1.0)[0] * self.request_weight
                else:
                    labs = active_domain_labels[domain].reshape(-1)
                    domain_scores = domain_scores[labs != -1]
                    labs = labs[labs != -1].float()
                    loss += self.active_domain_loss(domain_scores, labs) * self.active_domain_weight
    else:
        request_probs, domain_probs = None, None

    # Informable slot predictions
    inform_probs = dict()
    out_dict = dict()
    mutual_info = dict()
    stats = dict()
    for slot, slot_id in self.informable_slot_ids.items():
        # Get slot belief embedding and value candidates
        candidate_embeddings = getattr(self, slot + '_value_embeddings').to(turn_embeddings.device)
        belief = belief_embedding[:, :, slot_id, :]
        slot_size = candidate_embeddings.size(0)

        # Use similaroty matching to produce belief state
        if self.config.distance_measure in ['cosine', 'euclidean']:
            belief = belief.unsqueeze(2).repeat((1, 1, slot_size, 1))
            belief = belief.reshape(-1, self.config.hidden_size)

            # Pooling of set of value candidate description representation
            if self.config.set_similarity and self.config.set_pooling == 'mean':
                candidate_mask = (candidate_embeddings != 0.0).float()
                candidate_embeddings = candidate_embeddings.sum(1) / candidate_mask.sum(1)
            elif self.config.set_similarity and self.config.set_pooling == 'cnn':
                candidate_embeddings = candidate_embeddings.transpose(1, 2)
                candidate_embeddings = self.conv_pooler(candidate_embeddings).mean(-1)
            elif self.config.set_similarity and self.config.set_pooling == 'dan':
                candidate_mask = (candidate_embeddings != 0.0).float()
                candidate_embeddings = candidate_embeddings.sum(1) / torch.sqrt(torch.tensor(candidate_mask.sum(1)))
                candidate_embeddings = self.avg_net(candidate_embeddings)

            candidate_embeddings = candidate_embeddings.unsqueeze(0).unsqueeze(0).repeat((batch_size,
                                                                                          dialogue_size, 1, 1))
            candidate_embeddings = candidate_embeddings.reshape(-1, self.config.hidden_size)

        # Score value candidates
        if self.config.distance_measure == 'cosine':
            scores = self.distance(belief, candidate_embeddings)
            # *27 here rescales the cosine similarity for better learning
            scores = scores.reshape(batch_size * dialogue_size, -1) * 27.0
        elif self.config.distance_measure == 'euclidean':
            scores = -1.0 * self.distance(belief, candidate_embeddings)
            scores = scores.reshape(batch_size * dialogue_size, -1)

        # Calculate belief state
        if self.config.loss_function in ['crossentropy', 'inhibitedce',
                                         'labelsmoothing', 'distillation', 'distribution_distillation']:
            probs_ = torch.softmax(scores.reshape(batch_size, dialogue_size, -1), -1)
        elif self.config.loss_function in ['bayesianmatching']:
            probs_ = torch.sigmoid(scores.reshape(batch_size, dialogue_size, -1))

        # Compute knowledge uncertainty in the beleif states
        if calculate_inform_mutual_info and self.config.loss_function == 'distribution_distillation':
            mutual_info[slot] = logits_to_mutual_info(scores).reshape(batch_size, dialogue_size)

        # Set padded turn probabilities to zero
        mask = attention_mask[self.slot_ids[slot],:, 0].reshape(batch_size, dialogue_size)
        batches, dialogues = torch.where(mask == 0.0)
        probs_[batches, dialogues, :] = 0.0
        inform_probs[slot] = probs_

        # Calculate belief state loss
        if state_labels is not None and slot in state_labels:
            if self.config.loss_function == 'bayesianmatching':
                prior = torch.ones(scores.size(-1)).float().to(scores.device)
                prior = prior * self.config.prior_constant
                prior = prior.unsqueeze(0).repeat((scores.size(0), 1))

                loss += self.loss(scores, state_labels[slot].reshape(-1), prior=prior)
            elif self.config.loss_function == 'distillation':
                labels = state_labels[slot]
                labels = labels.reshape(-1, labels.size(-1))
                loss += self.loss(scores, labels, self.temp)
            elif self.config.loss_function == 'distribution_distillation':
                labels = state_labels[slot]
                labels = labels.reshape(-1, labels.size(-2), labels.size(-1))
                loss_, model_stats, ensemble_stats = self.loss(scores, labels, 1.0, 1.0)
                loss += loss_

                # Calculate stats regarding model precisions
                precision = model_stats['precision']
                ensemble_precision = ensemble_stats['precision']
                stats[slot] = {'model_precision_min': precision.min(),
                               'model_precision_max': precision.max(),
                               'model_precision_mean': precision.mean(),
                               'ensemble_precision_min': ensemble_precision.min(),
                               'ensemble_precision_max': ensemble_precision.max(),
                               'ensemble_precision_mean': ensemble_precision.mean()}
            else:
                loss += self.loss(scores, state_labels[slot].reshape(-1))

    # Return model outputs
    out = inform_probs, request_probs, domain_probs, goodbye_probs, context
    if state_labels is not None or request_labels is not None or active_domain_labels is not None or general_act_labels is not None:
        out = (loss,) + out + (stats,)
    if calculate_inform_mutual_info:
        out = out + (mutual_info,)
    return out


# Convert binary scores and labels to 2 class classification problem for distribution distillation
def convert_probs_to_logits(scores, labels):
    # Convert single target probability p to distribution [1-p, p]
    labels = labels.reshape(-1, labels.size(-1), 1)
    labels = torch.cat([1 - labels, labels], -1)

    # Convert input scores into predictive distribution [1-z, z]
    scores = torch.sigmoid(scores).unsqueeze(1)
    scores = torch.cat((1 - scores, scores), 1)
    scores = -1.0 * torch.log((1 / (scores + 1e-8)) - 1)  # Inverse sigmoid

    return scores, labels
