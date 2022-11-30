import os
import json
import copy
import logging

import torch
import transformers
from transformers import BertModel, BertConfig, BertTokenizer, RobertaModel, RobertaConfig, RobertaTokenizer

from convlab.dst.setsumbt.modeling import RobertaSetSUMBT, BertSetSUMBT
from convlab.dst.setsumbt.modeling.training import set_ontology_embeddings
from convlab.dst.dst import DST
from convlab.util.custom_util import model_downloader

USE_CUDA = torch.cuda.is_available()
transformers.logging.set_verbosity_error()


class SetSUMBTTracker(DST):
    """SetSUMBT Tracker object for Convlab dialogue system"""

    def __init__(self,
                 model_path: str = "",
                 model_type: str = "roberta",
                 return_turn_pooled_representation: bool = False,
                 return_confidence_scores: bool = False,
                 confidence_threshold='auto',
                 return_belief_state_entropy: bool = False,
                 return_belief_state_mutual_info: bool = False,
                 store_full_belief_state: bool = False):
        """
        Args:
            model_path: Model path or download URL
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
        self.model_path = model_path
        self.return_turn_pooled_representation = return_turn_pooled_representation
        self.return_confidence_scores = return_confidence_scores
        self.confidence_threshold = confidence_threshold
        self.return_belief_state_entropy = return_belief_state_entropy
        self.return_belief_state_mutual_info = return_belief_state_mutual_info
        self.store_full_belief_state = store_full_belief_state
        if self.store_full_belief_state:
            self.full_belief_state = {}
        self.info_dict = {}

        # Download model if needed
        if not os.path.exists(self.model_path):
            # Get path /.../convlab/dst/setsumbt/multiwoz/models
            download_path = os.path.dirname(os.path.abspath(__file__))
            download_path = os.path.join(download_path, 'models')
            if not os.path.exists(download_path):
                os.mkdir(download_path)
            model_downloader(download_path, self.model_path)
            # Downloadable model path format http://.../model_name.zip
            self.model_path = self.model_path.split('/')[-1].replace('.zip', '')
            self.model_path = os.path.join(download_path, self.model_path)

        # Select model type based on the encoder
        if model_type == "roberta":
            self.config = RobertaConfig.from_pretrained(self.model_path)
            self.tokenizer = RobertaTokenizer
            self.model = RobertaSetSUMBT
        elif model_type == "bert":
            self.config = BertConfig.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer
            self.model = BertSetSUMBT
        else:
            logging.debug("Name Error: Not Implemented")

        self.device = torch.device('cuda') if USE_CUDA else torch.device('cpu')

        self.load_weights()

    def load_weights(self):
        """Load model weights and model ontology"""
        logging.info('Loading SetSUMBT pretrained model.')
        self.tokenizer = self.tokenizer.from_pretrained(self.config.tokenizer_name)
        logging.info(f'Model tokenizer loaded from {self.config.tokenizer_name}.')
        self.model = self.model.from_pretrained(self.model_path, config=self.config)
        logging.info(f'Model loaded from {self.model_path}.')

        # Transfer model to compute device and setup eval environment
        self.model = self.model.to(self.device)
        self.model.eval()
        logging.info(f'Model transferred to device: {self.device}')

        logging.info('Loading model ontology')
        f = open(os.path.join(self.model_path, 'database', 'test.json'), 'r')
        self.ontology = json.load(f)
        f.close()

        db = torch.load(os.path.join(self.model_path, 'database', 'test.db'))
        set_ontology_embeddings(self.model, db)

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
        Update user actions and dialogue and belief states.

        Args:
            user_act:

        Returns:

        """
        prev_state = self.state
        _output = self.predict(self.get_features(user_act))

        # Format state entropy
        if _output[5] is not None:
            state_entropy = dict()
            for slot, e in _output[5].items():
                domain, slot = slot.split('-', 1)
                if domain not in state_entropy:
                    state_entropy[domain] = dict()
                state_entropy[domain][slot] = e
        else:
            state_entropy = None

        # Format state mutual information
        if _output[6] is not None:
            state_mutual_info = dict()
            for slot, mi in _output[6].items():
                domain, slot = slot.split('-', 1)
                if domain not in state_mutual_info:
                    state_mutual_info[domain] = dict()
                state_mutual_info[domain][slot] = mi[0, 0]
        else:
            state_mutual_info = None

        # Format all confidence scores
        belief_state_confidence = None
        if _output[4] is not None:
            belief_state_confidence = dict()
            belief_state_conf, request_probs, active_domain_probs, general_act_probs = _output[4]
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

        # Get new domain activation actions
        new_domains = [d for d, active in _output[1].items() if active]
        new_domains = [d for d in new_domains if not self.active_domains.get(d, False)]
        self.active_domains = _output[1]

        user_acts = _output[2]
        for domain in new_domains:
            user_acts.append(['inform', domain, 'none', 'none'])

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        for domain, substate in _output[0].items():
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

        if _output[3] is not None:
            new_state['turn_pooled_representation'] = _output[3]

        self.state = new_state
        self.info_dict = copy.deepcopy(dict(new_state))

        return self.state

    def predict(self, features: dict) -> tuple:
        """
        Model forward pass and prediction post processing.

        Args:
            features: Dictionary of model input features

        Returns:
            out: Model predictions and uncertainty features
        """
        state_mutual_info = None
        with torch.no_grad():
            turn_pooled_representation = None
            if self.return_turn_pooled_representation:
                _outputs = self.model(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'],
                                      attention_mask=features['attention_mask'], hidden_state=self.hidden_states,
                                      get_turn_pooled_representation=True)
                belief_state = _outputs[0]
                request_probs = _outputs[1]
                active_domain_probs = _outputs[2]
                general_act_probs = _outputs[3]
                self.hidden_states = _outputs[4]
                turn_pooled_representation = _outputs[5]
            elif self.return_belief_state_mutual_info:
                _outputs = self.model(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'],
                                      attention_mask=features['attention_mask'], hidden_state=self.hidden_states,
                                      get_turn_pooled_representation=True, calculate_state_mutual_info=True)
                belief_state = _outputs[0]
                request_probs = _outputs[1]
                active_domain_probs = _outputs[2]
                general_act_probs = _outputs[3]
                self.hidden_states = _outputs[4]
                state_mutual_info = _outputs[5]
            else:
                _outputs = self.model(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'],
                                      attention_mask=features['attention_mask'], hidden_state=self.hidden_states,
                                      get_turn_pooled_representation=False)
                belief_state, request_probs, active_domain_probs, general_act_probs, self.hidden_states = _outputs

        # Convert belief state into dialog state
        dialogue_state = dict()
        for slot, probs in belief_state.items():
            dom, slot = slot.split('-', 1)
            if dom not in dialogue_state:
                dialogue_state[dom] = dict()
            val = self.ontology[dom][slot]['possible_values'][probs[0, 0, :].argmax().item()]
            if val != 'none':
                dialogue_state[dom][slot] = val

        if self.store_full_belief_state:
            self.full_belief_state = belief_state

        # Obtain model output probabilities
        if self.return_confidence_scores:
            state_entropy = None
            if self.return_belief_state_entropy:
                state_entropy = {slot: probs[0, 0, :] for slot, probs in belief_state.items()}
                state_entropy = {slot: self.relative_entropy(p).item() for slot, p in state_entropy.items()}

            # Confidence score is the max probability across all not "none" values candidates.
            belief_state_conf = {slot: probs[0, 0, 1:].max().item() for slot, probs in belief_state.items()}
            _request_probs = {slot: p[0, 0].item() for slot, p in request_probs.items()}
            _active_domain_probs = {domain: p[0, 0].item() for domain, p in active_domain_probs.items()}
            _general_act_probs = {'bye': general_act_probs[0, 0, 1].item(), 'thank': general_act_probs[0, 0, 2].item()}
            confidence_scores = (belief_state_conf, _request_probs, _active_domain_probs, _general_act_probs)
        else:
            confidence_scores = None
            state_entropy = None

        # Construct request action prediction
        request_acts = [slot for slot, p in request_probs.items() if p[0, 0].item() > 0.5]
        request_acts = [slot.split('-', 1) for slot in request_acts]
        request_acts = [['request', domain, slot, '?'] for domain, slot in request_acts]

        # Construct active domain set
        active_domains = {domain: p[0, 0].item() > 0.5 for domain, p in active_domain_probs.items()}

        # Construct general domain action
        general_acts = general_act_probs[0, 0, :].argmax(-1).item()
        general_acts = [[], ['bye'], ['thank']][general_acts]
        general_acts = [[act, 'general', 'none', 'none'] for act in general_acts]

        user_acts = request_acts + general_acts

        out = (dialogue_state, active_domains, user_acts, turn_pooled_representation, confidence_scores)
        out += (state_entropy, state_mutual_info)
        return out

    def relative_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute relative entrop for a probability distribution

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
            if context[-1][0] != 'sys':
                system_act = ''
            else:
                system_act = context[-1][-1]
        else:
            system_act = ''

        # Tokenize dialog
        features = self.tokenizer.encode_plus(user_act, system_act, add_special_tokens=True,
                                              max_length=self.config.max_turn_len, padding='max_length',
                                              truncation='longest_first')

        input_ids = torch.tensor(features['input_ids']).reshape(
            1, 1, -1).to(self.device) if 'input_ids' in features else None
        token_type_ids = torch.tensor(features['token_type_ids']).reshape(
            1, 1, -1).to(self.device) if 'token_type_ids' in features else None
        attention_mask = torch.tensor(features['attention_mask']).reshape(
            1, 1, -1).to(self.device) if 'attention_mask' in features else None
        features = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

        return features


# if __name__ == "__main__":
#     from convlab.policy.vector.vector_uncertainty import VectorUncertainty
#     # from convlab.policy.vector.vector_binary import VectorBinary
#     tracker = SetSUMBTTracker(model_path='/gpfs/project/niekerk/src/SetSUMBT/models/SetSUMBT+ActPrediction-multiwoz21-roberta-gru-cosine-labelsmoothing-Seed0-10-08-22-12-42',
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
