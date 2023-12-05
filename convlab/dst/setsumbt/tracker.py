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
"""Run SetSUMBT belief tracker training and evaluation."""

import copy
import logging

import torch
import transformers

from convlab.dst.setsumbt.modeling import SetSUMBTModels
from convlab.dst.dst import DST

USE_CUDA = torch.cuda.is_available()
transformers.logging.set_verbosity_error()


class SetSUMBTTracker(DST):
    """SetSUMBT Tracker object for Convlab dialogue system"""

    def __init__(self,
                 model_name_or_path: str = "",
                 model_type: str = "roberta",
                 return_turn_pooled_representation: bool = False,
                 return_confidence_scores: bool = False,
                 confidence_threshold='auto',
                 return_belief_state_entropy: bool = False,
                 return_belief_state_mutual_info: bool = False,
                 store_full_belief_state: bool = True):
        """
        Args:
            model_name_or_path: Path to pretrained model or name of pretrained model
            model_type: Transformer type (roberta/bert)
            return_turn_pooled_representation: If true a turn level pooled representation is returned
            return_confidence_scores: If true act confidence scores are included in the state
            confidence_threshold: Confidence threshold value for constraints or option auto
            return_belief_state_entropy: If true belief state distribution entropies are included in the state
            return_belief_state_mutual_info: If true belief state distribution mutual infos are included in the state
            store_full_belief_state: If true full belief state is stored within tracker object
        """
        super(SetSUMBTTracker, self).__init__()

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.return_turn_pooled_representation = return_turn_pooled_representation
        self.return_confidence_scores = return_confidence_scores
        self.confidence_threshold = confidence_threshold
        self.return_belief_state_entropy = return_belief_state_entropy
        self.return_belief_state_mutual_info = return_belief_state_mutual_info
        self.store_full_belief_state = store_full_belief_state
        if self.store_full_belief_state:
            self.full_belief_state = {}
        self.info_dict = {}

        if self.model_type in SetSUMBTModels:
            self.model, _, self.config, self.tokenizer = SetSUMBTModels[self.model_type]
        else:
            raise NameError('NotImplemented')

        # Select model type based on the encoder
        self.config = self.config.from_pretrained(self.model_name_or_path)

        self.device = torch.device('cuda') if USE_CUDA else torch.device('cpu')
        self.load_weights()

    def load_weights(self):
        """Load model weights and model ontology"""
        logging.info('Loading SetSUMBT pretrained model.')
        self.tokenizer = self.tokenizer.from_pretrained(self.model_name_or_path)
        logging.info(f'Model tokenizer loaded from {self.model_name_or_path}.')
        self.model = self.model.from_pretrained(self.model_name_or_path, config=self.config)
        logging.info(f'Model loaded from {self.model_name_or_path}.')

        # Transfer model to compute device and setup eval environment
        self.model = self.model.to(self.device)
        self.model.eval()
        logging.info(f'Model transferred to device: {self.device}')

        logging.info('Loading model ontology')
        self.ontology = self.tokenizer.ontology

        if self.return_confidence_scores:
            logging.info('Model returns user action and belief state confidence scores.')
            self.get_thresholds(self.confidence_threshold)
            logging.info('Uncertain Querying set up and thresholds set up at:')
            logging.info(self.confidence_thresholds)
        if self.return_belief_state_entropy:
            logging.info('Model returns belief state distribution entropy scores (Total uncertainty).')
        if self.return_belief_state_mutual_info:
            logging.info('Model returns belief state distribution mutual information scores (Knowledge uncertainty).')
        logging.info('Ontology loaded successfully.')

    def get_thresholds(self, threshold='auto') -> dict:
        """
        Setup dictionary of domain specific confidence thresholds

        Args:
            threshold: Threshold value or option auto

        Returns:
            confidence_thresholds: Domain specific confidence thresholds
        """
        self.confidence_thresholds = dict()
        for domain, substate in self.ontology.items():
            for slot, slot_info in substate.items():
                # Auto thresholds are set based on the number of value candidates per slot
                if domain not in self.confidence_thresholds:
                    self.confidence_thresholds[domain] = dict()
                if threshold == 'auto':
                    thres = 1.0 / (float(len(slot_info['possible_values'])) - 2.1)
                    self.confidence_thresholds[domain][slot] = max(0.05, thres)
                else:
                    self.confidence_thresholds[domain][slot] = max(0.05, threshold)

        return self.confidence_thresholds

    def init_session(self):
        """Initialize dialogue state"""
        self.state = dict()
        self.state['belief_state'] = dict()
        self.state['booked'] = dict()
        for domain, substate in self.ontology.items():
            self.state['belief_state'][domain] = dict()
            for slot, slot_info in substate.items():
                if slot_info['possible_values'] and slot_info['possible_values'] != ['?']:
                    self.state['belief_state'][domain][slot] = ''
            self.state['booked'][domain] = list()
        self.state['history'] = []
        self.state['system_action'] = []
        self.state['user_action'] = []
        self.state['terminated'] = False
        self.active_domains = {}
        self.hidden_states = None
        self.info_dict = {}

    def update(self, user_act: str = '') -> dict:
        """
        Update dialogue state based on user utterance.

        Args:
            user_act: User utterance

        Returns:
            state: Dialogue state
        """
        prev_state = self.state
        outputs = self.predict(self.get_features(user_act))

        # Format state entropy
        if outputs.state_entropy is not None:
            state_entropy = dict()
            for slot, e in outputs.state_entropy.items():
                domain, slot = slot.split('-', 1)
                if domain not in state_entropy:
                    state_entropy[domain] = dict()
                state_entropy[domain][slot] = e
        else:
            state_entropy = None

        # Format state mutual information
        if outputs.belief_state_mutual_information is not None:
            state_mutual_info = dict()
            for slot, mi in outputs.belief_state_mutual_information.items():
                domain, slot = slot.split('-', 1)
                if domain not in state_mutual_info:
                    state_mutual_info[domain] = dict()
                state_mutual_info[domain][slot] = mi[0, 0]
        else:
            state_mutual_info = None

        # Format all confidence scores
        belief_state_confidence = None
        if outputs.confidence_scores is not None:
            belief_state_confidence = dict()
            belief_state_conf, request_probs, active_domain_probs, general_act_probs = outputs.confidence_scores
            for slot, p in belief_state_conf.items():
                domain, slot = slot.split('-', 1)
                if domain not in belief_state_confidence:
                    belief_state_confidence[domain] = dict()
                if slot not in belief_state_confidence[domain]:
                    belief_state_confidence[domain][slot] = dict()
                belief_state_confidence[domain][slot]['inform'] = p

            for slot, p in request_probs.items():
                domain, slot = slot.split('-', 1)
                if domain not in belief_state_confidence:
                    belief_state_confidence[domain] = dict()
                if slot not in belief_state_confidence[domain]:
                    belief_state_confidence[domain][slot] = dict()
                belief_state_confidence[domain][slot]['request'] = p

            for domain, p in active_domain_probs.items():
                if domain not in belief_state_confidence:
                    belief_state_confidence[domain] = dict()
                belief_state_confidence[domain]['none'] = {'inform': p}

            if 'general' not in belief_state_confidence:
                belief_state_confidence['general'] = dict()
            belief_state_confidence['general']['none'] = general_act_probs

        # Update belief state
        user_acts = outputs.state['user_action']

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        for domain, substate in outputs.state['belief_state'].items():
            for slot, value in substate.items():
                value = '' if value == 'none' else value
                value = 'dontcare' if value == 'do not care' else value
                value = 'guesthouse' if value == 'guest house' else value

                if domain not in new_belief_state:
                    if domain == 'bus':
                        continue
                    else:
                        logging.debug('Error: domain <{}> not in belief state'.format(domain))

                # Uncertainty clipping of state
                if belief_state_confidence is not None:
                    threshold = self.confidence_thresholds[domain][slot]
                    if belief_state_confidence[domain][slot].get('inform', 1.0) < threshold:
                        value = ''

                new_belief_state[domain][slot] = value
                if prev_state['belief_state'][domain][slot] != value:
                    user_acts.append(['inform', domain, slot, value])
                else:
                    bug = f'Unknown slot name <{slot}> with value <{value}> of domain <{domain}>'
                    logging.debug(bug)

        # Make all action domains active
        for domain in outputs.state['active_domains']:
            if domain in user_act.lower():
                outputs.state['active_domains'][domain] = True
        for intent, domain, slot, value in user_acts:
            outputs.state['active_domains'][domain] = True

        # Get new domain activation actions
        new_domains = [d for d, active in outputs.state['active_domains'].items() if active]
        new_domains = [d for d in new_domains if not self.active_domains.get(d, False)]
        self.active_domains = outputs.state['active_domains']

        for domain in new_domains:
            user_acts.append(['inform', domain, 'none', 'none'])

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['active_domains'] = self.active_domains
        if belief_state_confidence is not None:
            new_state['belief_state_probs'] = belief_state_confidence
        if state_entropy is not None:
            new_state['entropy'] = state_entropy
        if state_mutual_info is not None:
            new_state['mutual_information'] = state_mutual_info

        user_acts = [act for act in user_acts if act not in new_state['system_action']]
        new_state['user_action'] = user_acts

        if outputs.turn_pooled_representation is not None:
            new_state['turn_pooled_representation'] = outputs.turn_pooled_representation.reshape(-1)

        self.state = new_state
        # self.info_dict['belief_state'] = copy.deepcopy(dict(new_state))

        return self.state

    def predict(self, features: dict) -> tuple:
        """
        Model forward pass and prediction post-processing.

        Args:
            features: Dictionary of model input features

        Returns:
            out: Model predictions and uncertainty features
        """
        state_mutual_info = None
        with torch.no_grad():
            features['hidden_state'] = self.hidden_states
            features['get_turn_pooled_representation'] = self.return_turn_pooled_representation
            mutual_info = self.return_belief_state_mutual_info or self.store_full_belief_state
            features['calculate_state_mutual_info'] = mutual_info
            outputs = self.model(**features)
            self.hidden_states = outputs.hidden_state

        # Convert belief state into dialog state
        state = self.tokenizer.decode_state_batch(outputs.belief_state, outputs.request_probabilities,
                                                  outputs.active_domain_probabilities,
                                                  outputs.general_act_probabilities)
        state = state['000000'][0]

        if self.store_full_belief_state:
            self.info_dict['belief_state_distributions'] = outputs.belief_state
            self.info_dict['belief_state_knowledge_uncertainty'] = outputs.belief_state_mutual_information

        # Obtain model output probabilities
        if self.return_confidence_scores:
            state_entropy = None
            if self.return_belief_state_entropy:
                state_entropy = {slot: probs[0, 0, :] for slot, probs in outputs.belief_state.items()}
                state_entropy = {slot: self.relative_entropy(p).item() for slot, p in state_entropy.items()}

            # Confidence score is the max probability across all not "none" values candidates.
            belief_state_conf = {slot: probs[0, 0, 1:].max().item() for slot, probs in outputs.belief_state.items()}
            _request_probs = {slot: p[0, 0].item() for slot, p in outputs.request_probabilities.items()}
            _active_domain_probs = {domain: p[0, 0].item() for domain, p in outputs.active_domain_probabilities.items()}
            _general_act_probs = {'bye': outputs.general_act_probabilities[0, 0, 1].item(),
                                  'thank': outputs.general_act_probabilities[0, 0, 2].item()}
            confidence_scores = (belief_state_conf, _request_probs, _active_domain_probs, _general_act_probs)
        else:
            confidence_scores = None
            state_entropy = None

        outputs.confidence_scores = confidence_scores
        outputs.state_entropy = state_entropy
        outputs.state = state
        outputs.belief_state = None
        return outputs

    @staticmethod
    def relative_entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Compute relative entropy for a probability distribution

        Args:
            probs: Probability distributions

        Returns:
            entropy: Relative entropy
        """
        entropy = probs * torch.log(probs + 1e-8)
        entropy = -entropy.sum()
        # Maximum entropy of a K dimentional distribution is ln(K)
        entropy /= torch.log(torch.tensor(probs.size(-1)).float())

        return entropy

    def get_features(self, user_act: str) -> dict:
        """
        Tokenize utterances and construct model input features

        Args:
            user_act: User action string

        Returns:
            features: Model input features
        """
        # Extract system utterance from dialog history
        context = self.state['history']
        if context:
            sys_context = [utt for speaker, utt in context if speaker == 'sys']
            if sys_context:
                system_act = sys_context[-1]
            else:
                system_act = ''
        else:
            system_act = ''

        dialogue = [[{
            'user_utterance': user_act,
            'system_utterance': system_act
        }]]

        # Tokenize dialog
        features = self.tokenizer.encode(dialogue, max_seq_len=self.config.max_turn_len, max_turns=1)

        for key in features:
            if features[key] is not None:
                features[key] = features[key].to(self.device)

        return features


# if __name__ == "__main__":
#     from convlab.policy.vector.vector_uncertainty import VectorUncertainty
#     # from convlab.policy.vector.vector_binary import VectorBinary
#     tracker = SetSUMBTTracker(model_name_or_path='setsumbt_multiwoz21',
#                               return_confidence_scores=True, confidence_threshold='auto',
#                               return_belief_state_entropy=True)
#     vector = VectorUncertainty(use_state_total_uncertainty=True, confidence_thresholds=tracker.confidence_thresholds,
#                                use_masking=True)
#     # vector = VectorBinary()
#     tracker.init_session()
#
#     state = tracker.update('hey. I need a cheap restaurant.')
#     tracker.state['history'].append(['usr', 'hey. I need a cheap restaurant.'])
#     tracker.state['history'].append(['sys', 'There are many cheap places, which food do you like?'])
#     state = tracker.update('If you have something Asian that would be great.')
#     tracker.state['history'].append(['usr', 'If you have something Asian that would be great.'])
#     tracker.state['history'].append(['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
#     tracker.state['system_action'] = [['inform', 'restaurant', 'food', 'chinese'],
#                                       ['inform', 'restaurant', 'name', 'the golden wok']]
#     state = tracker.update('Great. Where are they located?')
#     tracker.state['history'].append(['usr', 'Great. Where are they located?'])
#     state = tracker.state
#     state['terminated'] = False
#     state['booked'] = {}
#
#     print(state)
#     print(vector.state_vectorize(state))
