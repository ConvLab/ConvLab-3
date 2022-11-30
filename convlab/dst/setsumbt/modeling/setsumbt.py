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
"""SetSUMBT Prediction Head"""

import torch
from torch.autograd import Variable
from torch.nn import (Module, MultiheadAttention, GRU, LSTM, Linear, LayerNorm, Dropout,
                      CosineSimilarity, CrossEntropyLoss, PairwiseDistance,
                      Sequential, ReLU, Conv1d, GELU, BCEWithLogitsLoss)
from torch.nn.init import (xavier_normal_, constant_)

from convlab.dst.setsumbt.loss import (BayesianMatchingLoss, BinaryBayesianMatchingLoss,
                                       KLDistillationLoss, BinaryKLDistillationLoss,
                                       LabelSmoothingLoss, BinaryLabelSmoothingLoss,
                                       RKLDirichletMediatorLoss, BinaryRKLDirichletMediatorLoss)


class SlotUtteranceMatching(Module):
    """Slot Utterance matching attention based information extractor"""

    def __init__(self, hidden_size: int = 768, attention_heads: int = 12):
        """
        Args:
            hidden_size (int): Dimension of token embeddings
            attention_heads (int): Number of attention heads to use in attention module
        """
        super(SlotUtteranceMatching, self).__init__()

        self.attention = MultiheadAttention(hidden_size, attention_heads)

    def forward(self,
                turn_embeddings: torch.Tensor,
                attention_mask: torch.Tensor,
                slot_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            turn_embeddings: Embeddings for each token in each turn [n_turns, turn_length, hidden_size]
            attention_mask: Padding mask for each turn [n_turns, turn_length, hidden_size]
            slot_embeddings: Embeddings for each token in the slot descriptions

        Returns:
            hidden: Information extracted from turn related to slot descriptions
        """
        turn_embeddings = turn_embeddings.transpose(0, 1)

        key_padding_mask = (attention_mask[:, :, 0] == 0.0)
        key_padding_mask[key_padding_mask[:, 0], :] = False

        hidden, _ = self.attention(query=slot_embeddings, key=turn_embeddings, value=turn_embeddings,
                                   key_padding_mask=key_padding_mask)

        attention_mask = attention_mask[:, 0, :].unsqueeze(0).repeat((slot_embeddings.size(0), 1, 1))
        hidden = hidden * attention_mask

        return hidden


class RecurrentNeuralBeliefTracker(Module):
    """Recurrent latent neural belief tracking module"""

    def __init__(self,
                 nbt_type: str = 'gru',
                 rnn_zero_init: bool = False,
                 input_size: int = 768,
                 hidden_size: int = 300,
                 hidden_layers: int = 1,
                 dropout_rate: float = 0.3):
        """
        Args:
            nbt_type: Type of recurrent neural network (gru/lstm)
            rnn_zero_init: Use zero initialised state for the RNN
            input_size: Embedding size of the inputs
            hidden_size: Hidden size of the RNN
            hidden_layers: Number of RNN Layers
            dropout_rate: Dropout rate
        """
        super(RecurrentNeuralBeliefTracker, self).__init__()

        if rnn_zero_init:
            self.belief_init = Sequential(Linear(input_size, hidden_size), ReLU(), Dropout(dropout_rate))
        else:
            self.belief_init = None

        self.nbt_type = nbt_type
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        if nbt_type == 'gru':
            self.nbt = GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=hidden_layers,
                           dropout=0.0 if hidden_layers == 1 else dropout_rate,
                           batch_first=True)
        elif nbt_type == 'lstm':
            self.nbt = LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=hidden_layers,
                            dropout=0.0 if hidden_layers == 1 else dropout_rate,
                            batch_first=True)
        else:
            raise NameError('Not Implemented')

        # Initialise Parameters
        xavier_normal_(self.nbt.weight_ih_l0)
        xavier_normal_(self.nbt.weight_hh_l0)
        constant_(self.nbt.bias_ih_l0, 0.0)
        constant_(self.nbt.bias_hh_l0, 0.0)

        # Intermediate feature mapping and layer normalisation
        self.intermediate = Linear(hidden_size, input_size)
        self.layer_norm = LayerNorm(input_size)
        self.dropout = Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Latent turn level information
            hidden_state: Latent internal belief state

        Returns:
            belief_embedding: Belief state embeddings
            context: Latent internal belief state
        """
        self.nbt.flatten_parameters()
        if hidden_state is None:
            if self.belief_init is None:
                context = torch.zeros(self.hidden_layers, inputs.size(0), self.hidden_size).to(inputs.device)
            else:
                context = self.belief_init(inputs[:, 0, :]).unsqueeze(0).repeat((self.hidden_layers, 1, 1))
            if self.nbt_type == "lstm":
                context = (context, torch.zeros(self.hidden_layers, inputs.size(0), self.hidden_size).to(inputs.device))
        else:
            context = hidden_state.to(inputs.device)

        # [batch_size, dialogue_size, nbt_hidden_size]
        belief_embedding, context = self.nbt(inputs, context)

        # Normalisation and regularisation
        belief_embedding = self.layer_norm(self.intermediate(belief_embedding))
        belief_embedding = self.dropout(belief_embedding)

        return belief_embedding, context


class SetPooler(Module):
    """Token set pooler"""

    def __init__(self, pooling_strategy: str = 'cnn', hidden_size: int = 768):
        """
        Args:
            pooling_strategy: Type of set pooler (cnn/dan/mean)
            hidden_size: Token embedding size
        """
        super(SetPooler, self).__init__()

        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'cnn':
            self.cnn_filter_size = 3
            self.pooler = Conv1d(hidden_size, hidden_size, self.cnn_filter_size)
        elif pooling_strategy == 'dan':
            self.pooler = Sequential(Linear(hidden_size, hidden_size), GELU(), Linear(2 * hidden_size, hidden_size))

    def forward(self, inputs, attention_mask):
        """
        Args:
            inputs: Token set embeddings
            attention_mask: Padding mask for the set of tokens

        Returns:

        """
        if self.pooling_strategy == "mean":
            hidden = inputs.sum(1) / attention_mask.sum(1)
        elif self.pooling_strategy == "cnn":
            hidden = self.pooler(inputs.transpose(1, 2)).mean(-1)
        elif self.pooling_strategy == 'dan':
            hidden = inputs.sum(1) / torch.sqrt(torch.tensor(attention_mask.sum(1)))
            hidden = self.pooler(hidden)

        return hidden


class SetSUMBTHead(Module):
    """SetSUMBT Prediction Head for Language Models"""

    def __init__(self, config):
        """
        Args:
            config (configuration): Model configuration class
        """
        super(SetSUMBTHead, self).__init__()
        self.config = config
        # Slot Utterance matching attention
        self.slot_utterance_matching = SlotUtteranceMatching(config.hidden_size, config.slot_attention_heads)

        # Latent context tracker
        self.nbt = RecurrentNeuralBeliefTracker(config.nbt_type, config.rnn_zero_init, config.hidden_size,
                                                config.nbt_hidden_size, config.nbt_layers, config.dropout_rate)

        # Set pooler for set similarity model
        if self.config.set_similarity:
            self.set_pooler = SetPooler(config.set_pooling, config.hidden_size)

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

    def add_slot_candidates(self, slot_candidates: tuple):
        """
        Add slots to the model ontology, the tuples should contain the slot embedding, informable value embeddings
        and a request indicator, if the informable value embeddings is None the slot is not informable and if
        the request indicator is false the slot is not requestable.

        Args:
            slot_candidates: Tuple containing slot embedding, informable value embeddings and a request indicator
        """
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

    def add_value_candidates(self, slot: str, value_candidates: torch.Tensor, replace: bool = False):
        """
        Add value candidates for a slot

        Args:
            slot: Slot name
            value_candidates: Value candidate embeddings
            replace: If true existing value candidates are replaced
        """
        embeddings = getattr(self, slot + '_value_embeddings')

        if embeddings.size(0) == 0 or replace:
            embeddings = value_candidates
        else:
            embeddings = torch.cat((embeddings, value_candidates.to(embeddings.device)), 0)

        setattr(self, slot + '_value_embeddings', embeddings)

    def forward(self,
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
                calculate_state_mutual_info: bool = False):
        """
        Args:
            turn_embeddings: Token embeddings in the current turn
            turn_pooled_representation: Pooled representation of the current dialogue turn
            attention_mask: Padding mask for the current dialogue turn
            batch_size: Number of dialogues in the batch
            dialogue_size: Number of turns in each dialogue
            hidden_state: Latent internal dialogue belief state
            state_labels: Dialogue state labels
            request_labels: User request action labels
            active_domain_labels: Current active domain labels
            general_act_labels: General user action labels
            calculate_state_mutual_info: Return mutual information in the dialogue state

        Returns:
            out: Tuple containing loss, predictive distributions, model statistics and state mutual information
        """
        hidden_size = turn_embeddings.size(-1)
        # Initialise loss
        loss = 0.0

        # General Action predictions
        general_act_probs = None
        if self.config.predict_actions:
            # General action prediction
            general_act_logits = self.general_act_gate(turn_pooled_representation.reshape(batch_size * dialogue_size,
                                                                                          hidden_size))

            # Compute loss for general action predictions (weighted loss)
            if general_act_labels is not None:
                if self.config.loss_function == 'distillation':
                    general_act_labels = general_act_labels.reshape(-1, general_act_labels.size(-1))
                    loss += self.general_act_loss(general_act_logits, general_act_labels,
                                                  self.temp) * self.general_act_weight
                elif self.config.loss_function == 'distribution_distillation':
                    general_act_labels = general_act_labels.reshape(-1, general_act_labels.size(-2),
                                                                    general_act_labels.size(-1))
                    loss += self.general_act_loss(general_act_logits, general_act_labels)[0] * self.general_act_weight
                else:
                    general_act_labels = general_act_labels.reshape(-1)
                    loss += self.general_act_loss(general_act_logits, general_act_labels) * self.general_act_weight

            # Compute general action probabilities
            general_act_probs = torch.softmax(general_act_logits, -1).reshape(batch_size, dialogue_size, -1)

        # Slot utterance matching
        num_slots = self.slot_embeddings.size(0)
        slot_embeddings = self.slot_embeddings.reshape(-1, hidden_size)
        slot_embeddings = slot_embeddings.unsqueeze(1).repeat((1, batch_size * dialogue_size, 1))
        slot_embeddings = slot_embeddings.to(turn_embeddings.device)

        if self.config.set_similarity:
            # Slot mask shape [num_slots * slot_len, batch_size * dialogue_size, 768]
            slot_mask = (slot_embeddings != 0.0).float()

        hidden = self.slot_utterance_matching(turn_embeddings, attention_mask, slot_embeddings)

        if self.config.set_similarity:
            hidden = hidden * slot_mask
        # Hidden layer shape [num_dials, num_slots, num_turns, 768]
        hidden = hidden.transpose(0, 1).reshape(batch_size, dialogue_size, slot_embeddings.size(0), -1).transpose(1, 2)

        # Latent context tracking
        # [batch_size * num_slots, dialogue_size, 768]
        hidden = hidden.reshape(batch_size * slot_embeddings.size(0), dialogue_size, -1)
        belief_embedding, hidden_state = self.nbt(hidden, hidden_state)

        belief_embedding = belief_embedding.reshape(batch_size, slot_embeddings.size(0),
                                                    dialogue_size, -1).transpose(1, 2)
        if self.config.set_similarity:
            belief_embedding = belief_embedding.reshape(batch_size, dialogue_size, num_slots, -1,
                                                        self.config.hidden_size)
        # [batch_size, dialogue_size, num_slots, *slot_desc_len, 768]

        # Pooling of the set of latent context representation
        if self.config.set_similarity:
            slot_mask = slot_mask.transpose(0, 1).reshape(batch_size, dialogue_size, num_slots, -1, hidden_size)
            belief_embedding = belief_embedding * slot_mask

            belief_embedding = self.set_pooler(belief_embedding.reshape(-1, slot_mask.size(-2), hidden_size),
                                               slot_mask.reshape(-1, slot_mask.size(-2), hidden_size))
            belief_embedding = belief_embedding.reshape(batch_size, dialogue_size, num_slots, -1)

        # Perform classification
        # Get padded batch, dialogue idx pairs
        batches, dialogues = torch.where(attention_mask[:, 0, 0].reshape(batch_size, dialogue_size) == 0.0)
        
        if self.config.predict_actions:
            # User request prediction
            request_probs = dict()
            for slot, slot_id in self.requestable_slot_ids.items():
                request_logits = self.request_gate(belief_embedding[:, :, slot_id, :])

                # Store output probabilities
                request_logits = request_logits.reshape(batch_size, dialogue_size)
                # Set request scores to 0.0 for padded turns
                request_logits[batches, dialogues] = 0.0
                request_probs[slot] = torch.sigmoid(request_logits)

                if request_labels is not None:
                    # Compute request gate loss
                    request_logits = request_logits.reshape(-1)
                    if self.config.loss_function == 'distillation':
                        loss += self.request_loss(request_logits, request_labels[slot].reshape(-1),
                                                  self.temp) * self.request_weight
                    elif self.config.loss_function == 'distribution_distillation':
                        loss += self.request_loss(request_logits, request_labels[slot])[0] * self.request_weight
                    else:
                        labs = request_labels[slot].reshape(-1)
                        request_logits = request_logits[labs != -1]
                        labs = labs[labs != -1].float()
                        loss += self.request_loss(request_logits, labs) * self.request_weight

            # Active domain prediction
            active_domain_probs = dict()
            for domain, slot_ids in self.domain_ids.items():
                belief = belief_embedding[:, :, slot_ids, :]
                if len(slot_ids) > 1:
                    # SqrtN reduction across all slots within a domain
                    belief = belief.sum(2) / ((belief != 0.0).float().sum(2) ** 0.5)
                active_domain_logits = self.active_domain_gate(belief)

                # Store output probabilities
                active_domain_logits = active_domain_logits.reshape(batch_size, dialogue_size)
                active_domain_logits[batches, dialogues] = 0.0
                active_domain_probs[domain] = torch.sigmoid(active_domain_logits)

                if active_domain_labels is not None and domain in active_domain_labels:
                    # Compute domain prediction loss
                    active_domain_logits = active_domain_logits.reshape(-1)
                    if self.config.loss_function == 'distillation':
                        loss += self.active_domain_loss(active_domain_logits, active_domain_labels[domain].reshape(-1),
                                                        self.temp) * self.active_domain_weight
                    elif self.config.loss_function == 'distribution_distillation':
                        loss += self.active_domain_loss(active_domain_logits,
                                                        active_domain_labels[domain])[0] * self.active_domain_weight
                    else:
                        labs = active_domain_labels[domain].reshape(-1)
                        active_domain_logits = active_domain_logits[labs != -1]
                        labs = labs[labs != -1].float()
                        loss += self.active_domain_loss(active_domain_logits, labs) * self.active_domain_weight
        else:
            request_probs, active_domain_probs = None, None

        # Dialogue state predictions
        belief_state_probs = dict()
        belief_state_mutual_info = dict()
        belief_state_stats = dict()
        for slot, slot_id in self.informable_slot_ids.items():
            # Get slot belief embedding and value candidates
            candidate_embeddings = getattr(self, slot + '_value_embeddings').to(turn_embeddings.device)
            belief = belief_embedding[:, :, slot_id, :]
            slot_size = candidate_embeddings.size(0)

            belief = belief.unsqueeze(2).repeat((1, 1, slot_size, 1))
            belief = belief.reshape(-1, self.config.hidden_size)

            if self.config.set_similarity:
                candidate_embeddings = self.set_pooler(candidate_embeddings, (candidate_embeddings != 0.0).float())
            candidate_embeddings = candidate_embeddings.unsqueeze(0).unsqueeze(0).repeat((batch_size,
                                                                                          dialogue_size, 1, 1))
            candidate_embeddings = candidate_embeddings.reshape(-1, self.config.hidden_size)

            # Score value candidates
            if self.config.distance_measure == 'cosine':
                logits = self.distance(belief, candidate_embeddings)
                # *27 here rescales the cosine similarity for better learning
                logits = logits.reshape(batch_size * dialogue_size, -1) * 27.0
            elif self.config.distance_measure == 'euclidean':
                logits = -1.0 * self.distance(belief, candidate_embeddings)
                logits = logits.reshape(batch_size * dialogue_size, -1)

            # Calculate belief state
            probs_ = torch.softmax(logits.reshape(batch_size, dialogue_size, -1), -1)

            # Compute knowledge uncertainty in the beleif states
            if calculate_state_mutual_info and self.config.loss_function == 'distribution_distillation':
                belief_state_mutual_info[slot] = self.loss.logits_to_mutual_info(logits).reshape(batch_size, dialogue_size)

            # Set padded turn probabilities to zero
            probs_[batches, dialogues, :] = 0.0
            belief_state_probs[slot] = probs_

            # Calculate belief state loss
            if state_labels is not None and slot in state_labels:
                if self.config.loss_function == 'bayesianmatching':
                    prior = torch.ones(logits.size(-1)).float().to(logits.device)
                    prior = prior * self.config.prior_constant
                    prior = prior.unsqueeze(0).repeat((logits.size(0), 1))

                    loss += self.loss(logits, state_labels[slot].reshape(-1), prior=prior)
                elif self.config.loss_function == 'distillation':
                    labels = state_labels[slot]
                    labels = labels.reshape(-1, labels.size(-1))
                    loss += self.loss(logits, labels, self.temp)
                elif self.config.loss_function == 'distribution_distillation':
                    labels = state_labels[slot]
                    labels = labels.reshape(-1, labels.size(-2), labels.size(-1))
                    loss_, model_stats, ensemble_stats = self.loss(logits, labels)
                    loss += loss_

                    # Calculate stats regarding model precisions
                    precision = model_stats['precision']
                    ensemble_precision = ensemble_stats['precision']
                    belief_state_stats[slot] = {'model_precision_min': precision.min(),
                                                'model_precision_max': precision.max(),
                                                'model_precision_mean': precision.mean(),
                                                'ensemble_precision_min': ensemble_precision.min(),
                                                'ensemble_precision_max': ensemble_precision.max(),
                                                'ensemble_precision_mean': ensemble_precision.mean()}
                else:
                    loss += self.loss(logits, state_labels[slot].reshape(-1))

        # Return model outputs
        out = belief_state_probs, request_probs, active_domain_probs, general_act_probs, hidden_state
        if state_labels is not None or request_labels is not None:
            out = (loss,) + out + (belief_state_stats,)
        if calculate_state_mutual_info:
            out = out + (belief_state_mutual_info,)
        return out
