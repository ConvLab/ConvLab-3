import os
import json
import copy
import logging

import torch
import transformers
from transformers import (BertModel, BertConfig, BertTokenizer,
                          RobertaModel, RobertaConfig, RobertaTokenizer)
from convlab.dst.setsumbt.modeling import (RobertaSetSUMBT,
                                            BertSetSUMBT)

<<<<<<<< HEAD:convlab/dst/setsumbt/multiwoz/Tracker.py
from convlab.dst.dst import DST
from convlab.util.multiwoz.state import default_state
from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab.dst.rule.multiwoz import normalize_value
from convlab.util.custom_util import model_downloader
========
from convlab2.dst.dst import DST
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab2.dst.rule.multiwoz import normalize_value
from convlab2.util.custom_util import model_downloader
from convlab2.dst.setsumbt.modeling.training import set_ontology_embeddings
>>>>>>>> setsumbt_unifiedformat:convlab2/dst/setsumbt/unified_format_data/Tracker.py

USE_CUDA = torch.cuda.is_available()

# Map from SetSUMBT slot names to Convlab slot names
SLOT_MAP = {'arrive by': 'arriveBy',
            'leave at': 'leaveAt',
            'price range': 'pricerange',
            'trainid': 'trainID',
            'reference': 'Ref',
            'taxi types': 'car type'}


class SetSUMBTTracker(DST):

    def __init__(self, model_path="", model_type="roberta",
                 get_turn_pooled_representation=False,
                 get_confidence_scores=False,
                 threshold='auto',
                 return_entropy=False,
                 return_mutual_info=False,
                 store_full_belief_state=False):
        super(SetSUMBTTracker, self).__init__()

        self.model_type = model_type
        self.model_path = model_path
        self.get_turn_pooled_representation = get_turn_pooled_representation
        self.get_confidence_scores = get_confidence_scores
        self.threshold = threshold
        self.return_entropy = return_entropy
        self.return_mutual_info = return_mutual_info
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
            # Downloadable model path format http://.../setsumbt_model_name.zip
            self.model_path = self.model_path.split('/')[-1].split('_', 1)[-1].replace('.zip', '')
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

        # Value dict for value normalisation
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

        self.load_weights()

    def load_weights(self):
        # Load tokenizer and model checkpoints
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
        if os.path.isfile(os.path.join(self.model_path, 'ontology.json')):
            ontology_dir = self.model_path
        else:
            ontology_dir = os.path.join(self.model_path, 'database')
        f = open(os.path.join(ontology_dir, 'test.json'), 'r')
        self.ontology = json.load(f)
        f.close()

        db = torch.load(os.path.join(ontology_dir, 'test.db'))
        set_ontology_embeddings(self.model, db)

        if self.get_confidence_scores:
            logging.info('Model will output action and state confidence scores.')
        if self.get_confidence_scores:
            self.get_thresholds(self.threshold)
            logging.info('Uncertain Querying set up and thresholds set up at:')
            logging.info(self.thresholds)
        if self.return_entropy:
            logging.info('Model will output state distribution entropy.')
        if self.return_mutual_info:
            logging.info('Model will output state distribution mutual information.')
        logging.info('Ontology loaded successfully.')

        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain

    #TODO
    def get_thresholds(self, threshold='auto'):
        self.thresholds = {}
        for slot, value_candidates in self.ontology.items():
            domain, slot = slot.split('-', 1)
            slot = REF_SYS_DA[domain.capitalize()].get(slot, slot)
            slot = slot.strip().split()[1] if 'book ' in slot else slot
            slot = SLOT_MAP.get(slot, slot)

            # Auto thresholds are set based on the number of value candidates per slot
            if domain not in self.thresholds:
                self.thresholds[domain] = {}
            if threshold == 'auto':
                thres = 1.0 / (float(len(value_candidates)) - 2.1)
                self.thresholds[domain][slot] = max(0.05, thres)
            else:
                self.thresholds[domain][slot] = max(0.05, threshold)

        return self.thresholds

    def init_session(self):
        self.state = {"belief_state": {domain: {slot: '' for slot, slot_info in substate.items()
                                                if slot_info['possible_values'] and slot_info['possible_values'] != ['?']}
                                       for domain, substate in self.ontology.items()}}
        self.state['history'] = []
        self.state['system_action'] = []
        self.state['user_action'] = []
        self.active_domains = {}
        self.hidden_states = None
        self.info_dict = {}

    def update(self, user_act=''):
        prev_state = self.state

        # Convert dialogs into transformer input features (token_ids, masks, etc)
        features = self.get_features(user_act)
        # Model forward pass
        pred_states, active_domains, user_acts, turn_pooled_representation, belief_state, entropy_, mutual_info_ = self.predict(
            features)

        if entropy_ is not None:
            entropy = {}
            for slot, e in entropy_.items():
                domain, slot = slot.split('-', 1)
                if domain not in entropy:
                    entropy[domain] = {}
                if 'book' in slot:
                    assert slot.startswith('book ')
                    slot = slot.strip().split()[1]
                slot = SLOT_MAP.get(slot, slot)
                entropy[domain][slot] = e
            del entropy_
        else:
            entropy = None

        if mutual_info_ is not None:
            mutual_info = {}
            for slot, mi in mutual_info_.items():
                domain, slot = slot.split('-', 1)
                if domain not in mutual_info:
                    mutual_info[domain] = {}
                if 'book' in slot:
                    assert slot.startswith('book ')
                    slot = slot.strip().split()[1]
                slot = SLOT_MAP.get(slot, slot)
                mutual_info[domain][slot] = mi[0, 0]
        else:
            mutual_info = None

        if belief_state is not None:
            bs_probs = {}
            belief_state, request_dist, domain_dist, greeting_dist = belief_state
            for slot, p in belief_state.items():
                domain, slot = slot.split('-', 1)
                if domain not in bs_probs:
                    bs_probs[domain] = {}
                if 'book' in slot:
                    assert slot.startswith('book ')
                    slot = slot.strip().split()[1]
                slot = SLOT_MAP.get(slot, slot)
                if slot not in bs_probs[domain]:
                    bs_probs[domain][slot] = {}
                bs_probs[domain][slot]['inform'] = p

            for slot, p in request_dist.items():
                domain, slot = slot.split('-', 1)
                if domain not in bs_probs:
                    bs_probs[domain] = {}
                slot = SLOT_MAP.get(slot, slot)
                if slot not in bs_probs[domain]:
                    bs_probs[domain][slot] = {}
                bs_probs[domain][slot]['request'] = p

            for domain, p in domain_dist.items():
                if domain not in bs_probs:
                    bs_probs[domain] = {}
                bs_probs[domain]['none'] = {'inform': p}

            if 'general' not in bs_probs:
                bs_probs['general'] = {}
            bs_probs['general']['none'] = greeting_dist

        new_domains = [d for d, active in active_domains.items() if active]
        new_domains = [
            d for d in new_domains if not self.active_domains.get(d, False)]
        self.active_domains = active_domains

        for domain in new_domains:
            user_acts.append(['Inform', domain.capitalize(), 'none', 'none'])

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        # user_acts = []
        for domain, substate in pred_states.items():
            for slot, value in substate.items():
                value = '' if value == 'none' else value
                value = 'dontcare' if value == 'do not care' else value
                value = 'guesthouse' if value == 'guest house' else value

                if domain not in new_belief_state:
                    if domain == 'bus':
                        continue
                    else:
                        logging.debug(
                            'Error: domain <{}> not in belief state'.format(domain))

                # Uncertainty clipping of state
                if belief_state is not None:
                    if bs_probs[domain][slot].get('inform', 1.0) < self.thresholds[domain][slot]:
                        value = ''

                new_belief_state[domain][slot] = value
                if prev_state['belief_state'][domain][slot] != value:
                    user_acts.append(['Inform', domain, slot, value])
                else:
                    logging.debug(
                        'unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(
                            slot, value, domain, state)
                    )

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['active_domains'] = self.active_domains
        if belief_state is not None:
            new_state['belief_state_probs'] = bs_probs
        if entropy is not None:
            new_state['entropy'] = entropy
        if mutual_info is not None:
            new_state['mutual_information'] = mutual_info

        user_acts = [act for act in user_acts if act not in new_state['system_action']]
        new_state['user_action'] = user_acts

        user_requests = [[a, d, s, v] for a, d, s, v in user_acts if a == 'Request']
        for act, domain, slot, value in user_requests:
            k = REF_SYS_DA[domain].get(slot, slot)
            domain = domain.lower()
            if domain not in new_state['request_state']:
                new_state['request_state'][domain] = {}
            if k not in new_state['request_state'][domain]:
                new_state['request_state'][domain][k] = 0

        if turn_pooled_representation is not None:
            new_state['turn_pooled_representation'] = turn_pooled_representation

        self.state = new_state
        self.info_dict = copy.deepcopy(dict(new_state))

        return self.state

    # Model prediction function

    def predict(self, features):
        # Forward Pass
        mutual_info = None
        with torch.no_grad():
            turn_pooled_representation = None
            if self.get_turn_pooled_representation:
                belief_state, request, domain, goodbye, self.hidden_states, turn_pooled_representation = self.model(input_ids=features['input_ids'],
                                                                                                                    token_type_ids=features[
                                                                                                                        'token_type_ids'],
                                                                                                                    attention_mask=features[
                                                                                                                        'attention_mask'],
                                                                                                                    hidden_state=self.hidden_states,
                                                                                                                    get_turn_pooled_representation=True)
            elif self.return_mutual_info:
                belief_state, request, domain, goodbye, self.hidden_states, mutual_info = self.model(input_ids=features['input_ids'],
                                                                                                     token_type_ids=features[
                                                                                                         'token_type_ids'],
                                                                                                     attention_mask=features[
                                                                                                         'attention_mask'],
                                                                                                     hidden_state=self.hidden_states,
                                                                                                     get_turn_pooled_representation=False,
                                                                                                     calculate_inform_mutual_info=True)
            else:
                belief_state, request, domain, goodbye, self.hidden_states = self.model(input_ids=features['input_ids'],
                                                                                        token_type_ids=features['token_type_ids'],
                                                                                        attention_mask=features['attention_mask'],
                                                                                        hidden_state=self.hidden_states,
                                                                                        get_turn_pooled_representation=False)

        # Convert belief state into dialog state
        predictions = {}
        for slot, state in belief_state.items():
            dom, slot = slot.split('-', 1)
            if dom not in predictions:
                predictions[dom] = {}
            pred = self.ontology[dom][slot]['possible_values'][state[0, 0, :].argmax().item()]
            if pred != 'none':
                predictions[dom][slot] = pred

        if self.store_full_belief_state:
            self.full_belief_state = belief_state

        # Obtain model output probabilities
        if self.get_confidence_scores:
            entropy = None
            if self.return_entropy:
                entropy = {slot: state[0, 0, :]
                           for slot, state in belief_state.items()}
                entropy = {slot: self.relative_entropy(
                    p).item() for slot, p in entropy.items()}

            # Confidence score is the max probability across all not "none" values candidates.
            belief_state = {slot: state[0, 0, 1:].max().item()
                            for slot, state in belief_state.items()}
            request_dist = {SLOT_MAP.get(
                slot, slot): p[0, 0].item() for slot, p in request.items()}
            domain_dist = {domain: p[0, 0].item()
                           for domain, p in domain.items()}
            greeting_dist = {'bye': goodbye[0, 0, 1].item(
            ), 'thank': goodbye[0, 0, 2].item()}
            belief_state = (belief_state, request_dist,
                            domain_dist, greeting_dist)
        else:
            belief_state = None
            entropy = None

        # Construct request action prediction
        request = [slot for slot, p in request.items() if p[0, 0].item() > 0.5]
        request = [slot.split('-', 1) for slot in request]
        request = [[domain, SLOT_MAP.get(slot, slot)]
                   for domain, slot in request]
        request = [['Request', domain.capitalize(), REF_USR_DA[domain.capitalize()].get(
            slot, slot), '?'] for domain, slot in request]

        # Construct active domain set
        domain = {domain: p[0, 0].item() > 0.5 for domain, p in domain.items()}

        # Construct general domain action
        goodbye = goodbye[0, 0, :].argmax(-1).item()
        goodbye = [[], ['bye'], ['thank']][goodbye]
        goodbye = [[act, 'general', 'none', 'none'] for act in goodbye]

        user_acts = request + goodbye

        return predictions, domain, user_acts, turn_pooled_representation, belief_state, entropy, mutual_info

    def relative_entropy(self, probs):
        entropy = probs * torch.log(probs + 1e-8)
        entropy = -entropy.sum()
        # Maximum entropy of a K dimentional distribution is ln(K)
        entropy /= torch.log(torch.tensor(probs.size(-1)).float())

        return entropy

    # Convert dialog turns into model features
    def get_features(self, user_act):
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


if __name__ == "__main__":
    tracker = SetSUMBTTracker(model_path='models/SetSUMBT-Acts-multiwoz21-10%-roberta-gru-cosine-labelsmoothing-Seed20222202-20-04-22-16-04')
    tracker.init_session()
    state = tracker.update('hey. I need a cheap restaurant.')
    tracker.state['history'].append(['usr', 'hey. I need a cheap restaurant.'])
    tracker.state['history'].append(['sys', 'There are many cheap places, which food do you like?'])
    state = tracker.update('If you have something Asian that would be great.')
    tracker.state['history'].append(['usr', 'If you have something Asian that would be great.'])
    tracker.state['history'].append(['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
    tracker.state['system_action'] = [['Inform', 'restaurant', 'food', 'chinese'],
                                      ['Inform', 'restaurant', 'name', 'the golden wok']]
    state = tracker.update('Great. Where are they located?')
    tracker.state['history'].append(['usr', 'Great. Where are they located?'])
    print(tracker.state)
