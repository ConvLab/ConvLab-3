from convlab.policy.lava.multiwoz.corpora_inference import BOS, EOS, PAD
from convlab.policy.lava.multiwoz.latent_dialog.enc2dec.decoders import DecoderRNN
from convlab.policy.lava.multiwoz.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from convlab.policy.lava.multiwoz.latent_dialog.utils import get_detokenize
from convlab.policy.lava.multiwoz.utils.nlp import normalize
from convlab.policy.lava.multiwoz.utils import util, delexicalize
from convlab.policy.lava.multiwoz import corpora_inference
from convlab.policy.lava.multiwoz.latent_dialog import domain
from convlab.policy.lava.multiwoz.latent_dialog.models_task import *
from convlab.policy import Policy
from convlab.util.file_util import cached_path
from convlab.util.multiwoz.state import default_state
# from convlab.util.multiwoz.dbquery import Database
from data.unified_datasets.multiwoz21.database import Database
from copy import deepcopy
import json
import os
import random
import tempfile
import zipfile

import numpy as np
import re
import torch
import torch.optim as optim
from nltk import word_tokenize
from torch.autograd import Variable
from collections import OrderedDict, defaultdict
import pickle
import pdb


TEACH_FORCE = 'teacher_forcing'
TEACH_GEN = 'teacher_gen'
GEN = 'gen'
GEN_VALID = 'gen_valid'

placeholder_re = re.compile(r'\[(\s*[\w_\s]+)\s*\]')
number_re = re.compile(
    r'.*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s$')
time_re = re.compile(
    r'((?:\d{1,2}[:]\d{2,3})|(?:\d{1,2} (?:am|pm)))', re.IGNORECASE)

REQ_TOKENS = {}
# DOMAIN_REQ_TOKEN = ['restaurant', 'hospital', 'hotel','attraction', 'train', 'police', 'taxi']
DOMAIN_REQ_TOKEN = ['taxi','restaurant', 'hospital', 'hotel','attraction','train','police']
ACTIVE_BS_IDX = [13, 30, 35, 61, 72, 91, 93] #indexes in the BS indicating if domain is active
NO_MATCH_DB_IDX = [-1, 0, -1, 6, 12, 18, -1] # indexes in DB pointer indicating 0 match is found for that domain, -1 mean that domain has no DB
REQ_TOKENS['attraction'] = ["[attraction_address]", "[attraction_name]", "[attraction_phone]", "[attraction_postcode]", "[attraction_reference]", "[attraction_type]"]
REQ_TOKENS['hospital'] = ["[hospital_address]", "[hospital_department]", "[hospital_name]", "[hospital_phone]", "[hospital_postcode]"] #, "[hospital_reference]"
REQ_TOKENS['hotel'] = ["[hotel_address]", "[hotel_name]", "[hotel_phone]", "[hotel_postcode]", "[hotel_reference]", "[hotel_type]"]
REQ_TOKENS['restaurant'] = ["[restaurant_name]", "[restaurant_address]", "[restaurant_phone]", "[restaurant_postcode]", "[restaurant_reference]"]
REQ_TOKENS['train'] = ["[train_id]", "[train_reference]"]
REQ_TOKENS['police'] = ["[police_address]", "[police_phone]",
                        "[police_postcode]"]  # "[police_name]",
REQ_TOKENS['taxi'] = ["[taxi_phone]", "[taxi_type]"]


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector

def addBookingPointer(state, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    # pprint(pointer_vector)
    return pointer_vector

def getTextBookingPointer(state):
    """Add information about availability of the booking option."""
    booked = []
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    booked.append('restaurant')

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    booked.append('hotel')

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    booked.append('train')

    if len(booked) == 0:
        booked.append("none")

    return booked

def get_relevant_domains(state):
    domains = []

    for domain in state.keys():
        # print("--", domain, "--")
        for slot, value in state[domain].items():
            if len(value) > 0:
                # print(slot, value)
                domains.append(domain)
                break
   
    # print(domains)
    return domains

def addDBPointer(state, db):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction',
               'train']  
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {}
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        constraints = [[slot, value] for slot, value in state[domain].items() if value] if domain in state else []
        entities = db.query(domain, constraints, topk=10)
        num_entities[domain] = len(entities)
        if len(entities) > 0:
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    return list(pointer_vector), db_results, num_entities

def addTextDBPointer(domains, state, db, use_booked=False):
    """Create database pointer for all related domains."""
    # domains = ['restaurant', 'hotel', 'attraction', 'train']
    db_results = {}
    num_entities = {}
    db_text = []

    # if active_domain in domains:
    # for domain in list(set(domains) & set(['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police'])):
    for domain in DOMAIN_REQ_TOKEN:
        if domain in domains:
            if domain in ["taxi", "police", "hospital"]:
                db_text.extend([domain, "1"])
            else:
                entities = db.query(domain, state[domain]['semi'].items())
                num_entities[domain] = len(entities)
                if len(entities) > 10:
                    str_n = "many"
                elif len(entities) == 0:
                    str_n = "none"
                else:
                    str_n = str(len(entities))
                db_text.extend([domain, str_n])

                if len(entities) > 0:
                    # fields = dbPointer.table_schema(domain)
                    # db_results[domain] = dict(zip(fields, entities[0]))
                    for ent in entities:
                        for k in KEY_MAP.keys():
                            if k in ent:
                                ent[KEY_MAP[k]] = ent[k]
                    db_results[domain] = entities
                # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
                # pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    if use_booked:
        booked = getTextBookingPointer(state)
        db_text = [BOS] + ["matches"] + db_text + ["booked"] + booked + [EOS]
    else:
        db_text = [BOS] + db_text + [EOS]

    return db_text , db_results, num_entities

def delexicaliseReferenceNumber(sent, state):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction',
               'train', 'taxi', 'hospital']  # , 'police']

    if state['history'][-1][0]=="sys":
        # print(state["booked"])
        for domain in domains:
            if state['booked'][domain]:
                for slot in state['booked'][domain][0]:
                    val = '[' + domain + '_' + slot + ']'
                    key = normalize(state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize(
                        "ref#" + state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

    return sent

def domain_mark_not_mentioned(state, active_domain):
    if active_domain not in ['hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel'] or active_domain is None:
        return

    for s in state[active_domain]:
        if state[active_domain][s] == '':
            state[active_domain][s] = 'not mentioned'

def mark_not_mentioned(state):
    for domain in state:
        # if domain == 'history':
        if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
            continue
        try:
            # if len([s for s in state[domain]['semi'] if s != 'book' and state[domain]['semi'][s] != '']) > 0:
            # for s in state[domain]['semi']:
            #     if s != 'book' and state[domain]['semi'][s] == '':
            #         state[domain]['semi'][s] = 'not mentioned'
            for s in state[domain]:
                if state[domain][s] == '':
                    state[domain][s] = 'not mentioned'
        except Exception as e:
            # print(str(e))
            # pprint(state[domain])
            pass

def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant',  u'hospital',
               u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate

def get_summary_bstate_unifiedformat(state):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant',  u'hospital',
               u'hotel', u'attraction', u'train']#, u'police']
    bstate = state['belief_state']
    # booked = state['booked']
    # how to make empty book this format instead of an empty dictionary?
    #TODO fix booked info update in state!
    booked = {
            "taxi": [],
            "hotel": [],
            "restaurant": [],
            "train": [],
            "attraction": [],
            "hospital": []
            }

    summary_bstate = []

    for domain in domains:
        domain_active = False

        booking = []
        if len(booked[domain]) > 0:
            booking.append(1)
        else:
            booking.append(0)
        if domain == 'train':
            if not bstate[domain]['book people']:
                booking.append(0)
            else:
                booking.append(1)
            if booked[domain] and 'ticket' in booked[domain][0].keys():
                booking.append(1)
            else:
                booking.append(0)
        summary_bstate += booking

        if domain == "restaurant":
            book_slots = ['book day', 'book people', 'book time']
        elif domain == "hotel":
            book_slots = ['book day', 'book people', 'book stay']
        else:
            book_slots = []
        for slot in book_slots:
            if bstate[domain][slot] == '':
                summary_bstate.append(0)
            else:
                summary_bstate.append(1)

        for slot in [s for s in bstate[domain] if "book" not in s]:
            slot_enc = [0, 0, 0]
            if bstate[domain][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain][slot] == 'dont care' or bstate[domain][slot] == 'dontcare' or bstate[domain][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active: # 7 domains
            summary_bstate += [1]
        else:
            summary_bstate += [0]


    # add manually from action as police is not tracked anymore in unified format
    if "Police" in [act[1] for act in state['user_action']]:
        summary_bstate += [0, 1]
    else:
        summary_bstate += [0, 0]

    assert len(summary_bstate) == 94
    return summary_bstate


DEFAULT_CUDA_DEVICE = -1


class LAVA(Policy):
    def __init__(self,
                 model_file="", is_train=False):

        if not model_file:
            raise Exception("No model for LAVA is specified!")

        temp_path = os.path.dirname(os.path.abspath(__file__))

        self.prev_state = default_state()
        self.prev_active_domain = None

        domain_name = 'object_division'
        domain_info = domain.get_domain(domain_name)
        self.db=Database()
       
        path, _ = os.path.split(model_file)
        if "rl-" in model_file:
            rl_config_path = os.path.join(path, "rl_config.json")
            self.rl_config = Pack(json.load(open(rl_config_path)))
            config_path = os.path.join(
                os.path.split(path)[:-1][0], "config.json")
        else:
            config_path = os.path.join(path, "config.json")
        config = Pack(json.load(open(config_path)))

        config.use_gpu = config.use_gpu and torch.cuda.is_available()
        try:
            self.corpus = corpora_inference.NormMultiWozCorpus(config)
        except (FileNotFoundError, PermissionError):
            train_data_path = "/gpfs/project/lubis/NeuralDialog-LaRL/data/norm-multi-woz/train_dials.json"
            config['train_path'] = train_data_path
            config['valid_path'] = train_data_path.replace("train", "val") 
            config['test_path'] = train_data_path.replace("train", "test") 
            self.corpus = corpora_inference.NormMultiWozCorpus(config)

        if "rl" in model_file:
            if "gauss" in model_file:
                self.model = SysPerfectBD2Gauss(self.corpus, config)
            else:
                if "e2e" in model_file:
                    self.model = SysE2ECat(self.corpus, config)
                else:
                    self.model = SysPerfectBD2Cat(self.corpus, config)
        else:
            if "actz" in model_file:
                if "gauss" in model_file:
                    self.model = SysActZGauss(self.corpus, config)
                else:
                    self.model = SysActZCat(self.corpus, config)
            elif "mt" in model_file:
                if "gauss" in model_file:
                    self.model = SysMTGauss(self.corpus, config)
                else:
                    self.model = SysMTCat(self.corpus, config)
            else:
                if "gauss" in model_file:
                    self.model = SysPerfectBD2Gauss(self.corpus, config)
                else:
                    self.model = SysPerfectBD2Cat(self.corpus, config)

        self.config = config
        self.input_lang_word2index = self.model.vocab_dict
        self.input_lang_index2word = {v:k for (k, v) in self.input_lang_word2index.items()}
        self.output_lang_word2index = self.input_lang_word2index
        self.output_lang_index2word = self.input_lang_index2word

        self.is_train = is_train
        if self.is_train:
            self.rl_config["US_best_reward_model_path"] = model_file.replace(
                ".model", "_US.model")
            if "lr_rl" not in config:
                self.config["lr_rl"] = 0.01
                self.config["gamma"] = 0.99

            tune_pi_only = True
            self.all_rewards = []
            self.logprobs = None
            self.opt = optim.SGD(
                [p for n, p in self.model.named_parameters(
                ) if 'c2z' in n or not tune_pi_only],
                lr=self.config.lr_rl,
                momentum=self.config.momentum,
                nesterov=False)

        if config.use_gpu:
            self.model.load_state_dict(torch.load(model_file))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(
                model_file, map_location=lambda storage, loc: storage))
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()
        self.dic = pickle.load(
            open(os.path.join(temp_path, 'lava_model/svdic.pkl'), 'rb'))
        self.fail_info_penalty = 0
        self.wrong_domain_penalty = 0
        self.num_generated_response = 0

    def reset():
        self.prev_state = default_state()

    def input_index2word(self, index):
        # if self.input_lang_index2word.has_key(index):
        if index in self.input_lang_index2word:
            return self.input_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def output_index2word(self, index):
        # if self.output_lang_index2word.has_key(index):
        if index in self.output_lang_index2word:
            return self.output_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def input_word2index(self, index):
        # if self.input_lang_word2index.has_key(index):
        if index in self.input_lang_word2index:
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        # if self.output_lang_word2index.has_key(index):
        if index in self.output_lang_word2index:
            return self.output_lang_word2index[index]
        else:
            return 2

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)),
                         dtype,
                         self.config.use_gpu)

    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        for b_id in range(context.shape[0]):
            utts.append(context[b_id, context_lens[b_id]-1])
        return np.array(utts)

    def is_masked_action(self, bs_label, db_label, response):
        """
        check if the generated response should be masked based on belief state and db result
        a) inform when there is no db match
        b) inform no option when there is a match
        c) out of domain action
        d) no offer with a particular slot?
        e) inform/request time on domains other than train and restaurant
        """
        for domain, bs_idx, db_idx in zip(DOMAIN_REQ_TOKEN, ACTIVE_BS_IDX, NO_MATCH_DB_IDX):
            if bs_label[bs_idx] == 0: # if domain is inactive
                if any([p in response for p in REQ_TOKENS[domain]]): # but a token from that domain is present
                    # print("MASK: inactive domain {} is mentioned".format(domain))
                    return True
            else: # domain is active
                if any([p in response for p in ["sorry", "no", "not" "cannot"]]): # system inform no offer
                    if db_idx < 0: # domain has no db
                        # print("MASK: inform no offer on domain {} without DB".format(domain))
                        return True
                    elif  db_label[db_idx] != 1: # there are matches
                        # print("MASK: inform no offer when there are matches on domain {}".format(domain))
                        return True
                    # if "[value_" in response:
                        # print("MASK: inform no offer mentioning criteria")
                        # return True # always only inform "no match for your criteria" w/o mentioning them explicitly
                elif any([p in response for p in REQ_TOKENS[domain]]) or "i have [value_count]" in response or "there are [value_count]" in response: # if requestable token is present
                    if db_idx >= 0 and int(db_label[db_idx]) == 1: # and domain has a DB to be queried and there are no matches
                        # print("MASK: inform match when there are no DB match on domain {}".format(domain))
                        return True

        return False

    def is_active(self, domain, state):

        if domain in [act[1] for act in state['user_action']]:
            return True
        else:
            return False

    def get_active_domain_unified(self, prev_active_domain, prev_state, state):
        domains = ['hotel', 'restaurant', 'attraction',
                   'train', 'taxi', 'hospital', 'police']
        active_domain = None
        # print("PREV_STATE:",prev_state)
        # print()
        # print("NEW_STATE",state)
        # print()
        for domain in domains:
            if not self.is_active(domain, prev_state) and self.is_active(domain, state):
                #print("case 1:",domain)
                return domain
            elif self.is_active(domain, prev_state) and self.is_active(domain, state):
                return domain
            # elif self.is_active(domain, prev_state) and not self.is_active(domain, state):
                #print("case 2:",domain)
                # return domain
            # elif prev_state['belief_state'][domain] != state['belief_state'][domain]:
                #print("case 3:",domain)
                # active_domain = domain
        if active_domain is None:
            active_domain = prev_active_domain
        return active_domain

    def predict(self, state):
        try:
            response, active_domain = self.predict_response(state)
        except Exception as e:
           print('Response generation error', e)
           response, active_domain = self.predict_response(state)
           response = 'Can I help you with anything else?'
           active_domain = None

        self.prev_state = deepcopy(state)
        self.prev_active_domain = active_domain

        return response

    def need_multiple_results(self, template):
        words = template.split()
        if "first" in words and "last" in words:
            return True
        elif "i have" in template:
            if words.count("[restaurant_name]") > 1:
                return True
            elif words.count("[restaurant_pricerange]") > 1:
                return True
            elif words.count("[hotel_name]") > 1:
                return True
            elif words.count("[attraction_name]") > 1:
                return True
            elif words.count("[train_id]") > 1:
                return True
            else:
                return False
        else:
            return False

    def predict_response (self, state):
        # input state is in convlab format
        history = []
        for i in range(len(state['history'])):
            history.append(state['history'][i][1])

        e_idx = len(history)
        s_idx = max(0, e_idx - self.config.backward_size)
        context = []
        for turn in history[s_idx: e_idx]:
            # turn = pad_to(config.max_utt_len, turn, do_pad=False)
            context.append(turn)

        # if len(state['history']) == 1:
        # corrected state reset
        if state['history'][s_idx][1] == "null":
            self.prev_state = default_state()
            self.prev_active_domain = None
            self.prev_output = ""
            self.domains = []

        prepared_data = {}
        prepared_data['context'] = []
        prepared_data['response'] = {}

        prev_action = deepcopy(self.prev_state['user_action'])
        prev_bstate = deepcopy(self.prev_state['belief_state'])
        state_history = state['history']
        action = deepcopy(state['user_action'])
        bstate = deepcopy(state['belief_state'])

        # mark_not_mentioned(prev_state)
        #active_domain = self.get_active_domain_convlab(self.prev_active_domain, prev_bstate, bstate)
        active_domain = self.get_active_domain_unified(self.prev_active_domain, self.prev_state, state)
        # print("---------")
        # print("active domain: ", active_domain)
        # if active_domain is not None:
            # print(f"DST on {active_domain}: {bstate[active_domain]}")

        domain_mark_not_mentioned(bstate, active_domain)

        top_results, num_results = None, None
        for t_id in range(len(context)):
            usr = context[t_id]
            # print(usr)

            if t_id == 0: #system turns
                if usr == "null":
                    usr = "<d>"
                    # booked = {"taxi": [],
                            # "restaurant": [],
                            # "hospital": [],
                            # "hotel": [],
                            # "attraction": [],
                            # "train": []}
            words = usr.split()

            usr = delexicalize.delexicalise(' '.join(words).lower(), self.dic)

            # parsing reference number GIVEN belief state
            usr = delexicaliseReferenceNumber(usr, state)

            # changes to numbers only here
            digitpat = re.compile('(^| )\d+( |$)')
            usr = re.sub(digitpat, '\\1[value_count]\\2', usr)
            
            # add database pointer
            pointer_vector, top_results, num_results = addDBPointer(bstate,self.db)
            if state['history'][-1][0] == "sys":
                booked = state['booked']
            #print(top_results)

            # add booking pointer
            pointer_vector = addBookingPointer(bstate, pointer_vector)
            belief_summary = get_summary_bstate_unifiedformat(state)

            usr_utt = [BOS] + usr.split() + [EOS]
            packed_val = {}
            packed_val['bs'] = belief_summary
            packed_val['db'] = pointer_vector
            packed_val['utt'] = self.corpus._sent2id(usr_utt)

            prepared_data['context'].append(packed_val)
        # if active_domain in ["restaurant", "hotel", "train", "attraction"]:
            # print(f"BS on {active_domain}: {num_results[active_domain]}")

        prepared_data['response']['bs'] = prepared_data['context'][-1]['bs']
        prepared_data['response']['db'] = prepared_data['context'][-1]['db']
        results = [Pack(context=prepared_data['context'],
                        response=prepared_data['response'])]

        # data_feed is in LaRL format
        data_feed = prepare_batch_gen(results, self.config)
        # print(belief_summary)

        for i in range(1):
            outputs = self.model_predict(data_feed)
            self.prev_output = outputs
            mul = False
            
            # default lexicalization
            if active_domain is not None and active_domain in num_results:
                num_results = num_results[active_domain]
            else:
                num_results = 0
            
            if active_domain is not None and active_domain in top_results:
                # detect response that requires multiple DB results
                if self.need_multiple_results(outputs):
                    # print("need multiple results")
                    mul = True
                    top_results = {active_domain: top_results[active_domain]}
                else:
                    if active_domain == 'train': #special case, where we want the last match instead of the first
                        if bstate['train']['arrive by'] != "not mentioned" and len(bstate['train']['arrive by']) > 0:
                            top_results = {active_domain: top_results[active_domain][-1]} # closest to arrive by
                        else:
                            top_results = {active_domain: top_results[active_domain][0]}
                    else:
                        top_results = {active_domain: top_results[active_domain][0]} # if active domain is wrong, this becomes the wrong entity
            else:
                top_results = {}
            state_with_history = deepcopy(bstate)
            state_with_history['history'] = deepcopy(state_history)

            if active_domain in ["hotel", "attraction", "train", "restaurant"] and active_domain not in top_results.keys(): # no db match for active domain
                if any([p in outputs for p in REQ_TOKENS[active_domain]]):
                    response = "I am sorry, can you say that again?"
                    database_results = {}
                else:
                    response = self.populate_template_unified(
                            outputs, top_results, num_results, state_with_history, active_domain)
                    # print(response)

            else:
                if mul and num_results > 1:
                    response = self.populate_template_options(outputs, top_results, num_results, state_with_history)
                else:
                    try:
                        response = self.populate_template_unified(
                        outputs, top_results, num_results, state_with_history, active_domain)
                    except:
                        print("can not lexicalize: ", outputs)
                        response = "I am sorry, can you say that again?"
                       
        response = response.replace("free pounds", "free")
        response = response.replace("pounds pounds", "pounds")
        if any([p in response for  p in ["not mentioned", "dontcare", "[", "]"]]):
            response = "I am sorry, can you say that again?"


        return response, active_domain


    def populate_template_unified(self, template, top_results, num_results, state, active_domain):
        # print("template:",template)
        # print("top_results:",top_results)
        # active_domain = None if len(
        #    top_results.keys()) == 0 else list(top_results.keys())[0]

        template = template.replace(
            'book [value_count] of', 'book one of')
        tokens = template.split()
        response = []
        for index, token in enumerate(tokens):
            if token.startswith('[') and (token.endswith(']') or token.endswith('].') or token.endswith('],')):
                domain = token[1:-1].split('_')[0]
                slot = token[1:-1].split('_')[1]
                if slot.endswith(']'):
                    slot = slot[:-1]
                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif active_domain != 'train' and slot == 'price':
                    slot = 'price range'
                elif slot == 'reference':
                    slot = 'Ref'
                if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if "there are" in " ".join(tokens[index-2:index]) or "i have" in " ".join(tokens[index-2:index]):
                            response.append(str(num_results))
                        # the first [value_count], the last [value_count]
                        elif "the" in tokens[index-2]:
                            response.append("one")
                        elif active_domain == "restaurant":
                            if "people" in tokens[index:index+1] or "table" in tokens[index-2:index]:
                                response.append(
                                    state[active_domain]["book people"])
                        elif active_domain == "train":
                            if "ticket" in " ".join(tokens[index-2:index+1]) or "people" in tokens[index:]:
                                response.append(
                                    state[active_domain]["book people"])
                            elif index+1 < len(tokens) and "minute" in tokens[index+1]:
                                response.append(
                                    top_results['train']['duration'].split()[0])
                        elif active_domain == "hotel":
                            if index+1 < len(tokens):
                                if "star" in tokens[index+1]:
                                    response.append(top_results['hotel']['stars'])
                                elif "nights" in tokens[index+1]:
                                    response.append(
                                        state[active_domain]["book stay"])
                                elif "people" in tokens[index+1]:
                                    response.append(
                                        state[active_domain]["book people"])
                        elif active_domain == "attraction":
                            if index + 1 < len(tokens):
                                if "pounds" in tokens[index+1] and "entrance fee" in " ".join(tokens[index-3:index]):
                                    value = top_results[active_domain]['entrance fee']
                                    if "?" in value:
                                        value = "unknown"
                                    # if "?" not in value:
                                    #    try:
                                    #        value = str(int(value))
                                    #    except:
                                    #        value = 'free'
                                    # else:
                                    #    value = "unknown"
                                    response.append(value)
                        # if "there are" in " ".join(tokens[index-2:index]):
                            # response.append(str(num_results))
                        # elif "the" in tokens[index-2]: # the first [value_count], the last [value_count]
                            # response.append("1")
                        else:
                            response.append(str(num_results))
                    elif slot == 'place':
                        if 'arriv' in " ".join(tokens[index-2:index]) or "to" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                response.append(
                                    top_results[active_domain]["destination"])
                            elif active_domain == "taxi":
                                response.append(
                                    state[active_domain]["destination"])
                        elif 'leav' in " ".join(tokens[index-2:index]) or "from" in tokens[index-2:index] or "depart" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                response.append(
                                    top_results[active_domain]["departure"])
                            elif active_domain == "taxi":
                                response.append(
                                    state[active_domain]["departure"])
                        elif "hospital" in template:
                            response.append("Cambridge")
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['destination', 'departure']:
                                        if s in state[d]:
                                            response.append(
                                                state[d][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'time':
                        if 'arrive' in ' '.join(response[-5:]) or 'arrival' in ' '.join(response[-5:]) or 'arriving' in ' '.join(response[-3:]):
                            if active_domain == "train" and 'arriveBy' in top_results[active_domain]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(
                                    top_results[active_domain]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arrive by' in state[d]:
                                    response.append(
                                        state[d]['arrive by'])
                                    break
                        elif 'leave' in ' '.join(response[-5:]) or 'leaving' in ' '.join(response[-5:]) or 'departure' in ' '.join(response[-3:]):
                            if active_domain == "train" and 'leaveAt' in top_results[active_domain]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(
                                    top_results[active_domain]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leave at' in state[d]:
                                    response.append(
                                        state[d]['leave at'])
                                    break
                        elif 'book' in response or "booked" in response:
                            if state['restaurant']['book time'] != "":
                                response.append(
                                    state['restaurant']['book time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arrive by', 'leave at']:
                                        if s in state[d]:
                                            response.append(
                                                state[d][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price':
                        if active_domain == 'attraction':
                            # .split()[0]
                            value = top_results['attraction']['entrance fee']
                            if "?" in value:
                                value = "unknown"
                            # if "?" not in value:
                            #    try:
                            #        value = str(int(value))
                            #    except:
                            #        value = 'free'
                            # else:
                            #    value = "unknown"
                            response.append(value)
                        elif active_domain == "train":
                            response.append(
                                top_results[active_domain][slot].split()[0])
                    elif slot == "day" and active_domain in ["restaurant", "hotel"]:
                        if state[active_domain]['book day'] != "":
                            response.append(
                                state[active_domain]['book day'])

                    else:
                        # slot-filling based on query results
                        for d in top_results:
                            if slot in top_results[d]:
                                response.append(top_results[d][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]:
                                    response.append(state[d][slot])
                                    break
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'phone':
                            response.append('01223216297')
                        elif slot == 'department':
                            response.append('neurosciences critical care unit')
                        elif slot == 'address':
                            response.append("56 Lincoln street")
                        elif slot == "postcode":
                            response.append('cb1p3')
                    elif domain == 'police':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'name':
                            response.append('Parkside Police Station')
                        elif slot == 'address':
                            response.append('Parkside, Cambridge')
                        elif slot == 'postcode':
                            response.append('cb3l3')
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'color':
                            # response.append(random.choice(["black","white","red","yellow","blue",'grey']))
                            response.append("black")
                        elif slot == 'type':
                            # response.append(random.choice(["toyota","skoda","bmw",'honda','ford','audi','lexus','volvo','volkswagen','tesla']))
                            response.append("toyota")
                    else:
                        # print(token)
                        response.append(token)
            else:
                if token == "pounds" and ("pounds" in response[-1] or "unknown" in response[-1] or "free" in response[-1]):
                    pass
                else:
                    response.append(token)

        try:
            response = ' '.join(response)
        except Exception as e:
            # pprint(response)
            raise
        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')

        # if "not mentioned" in response:
        #    pdb.set_trace()
        # print("lexicalized: ", response)

        return response

    def populate_template_options(self, template, top_results, num_results, state):
        # print("template:",template)
        # print("top_results:",top_results)
        active_domain = None if len(
            top_results.keys()) == 0 else list(top_results.keys())[0]
        # if active_domain != "train":
        #    pdb.set_trace()

        template = template.replace(
            'book [value_count] of', 'book one of')
        tokens = template.split()
        response = []
        result_idx = 0
        for index, token in enumerate(tokens):
            if token.startswith('[') and (token.endswith(']') or token.endswith('].') or token.endswith('],')):
                if "first" in tokens[index - 4:index]:
                    result_idx = 0
                elif "last" in tokens[index - 4:index] or "latest" in tokens[index-4:index]:
                    # pdb.set_trace()
                    result_idx = -1
                # this token has appeared before
                elif "name" in token and tokens[:index+1].count(token) > 1:
                    result_idx += 1
                domain = token[1:-1].split('_')[0]
                slot = token[1:-1].split('_')[1]
                if slot.endswith(']'):
                    slot = slot[:-1]
                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif active_domain != 'train' and slot == 'price':
                    slot = 'pricerange'
                elif slot == 'reference':
                    slot = 'Ref'
                if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain][result_idx]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][result_idx][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if "there are" in " ".join(tokens[index-2:index]) or "i have" in " ".join(tokens[index-2:index]):
                            response.append(str(num_results))
                        # the first [value_count], the last [value_count]
                        elif "the" in tokens[index-2] or "which" in tokens[index-1]:
                            response.append("one")
                        elif active_domain == "train":
                            if index+1 < len(tokens) and "minute" in tokens[index+1]:
                                response.append(
                                    top_results['train'][result_idx]['duration'].split()[0])
                        elif active_domain == "hotel":
                            if index+1 < len(tokens):
                                if "star" in tokens[index+1]:
                                    response.append(
                                        top_results['hotel'][result_idx]['stars'])
                                # elif "nights" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["stay"])
                                # elif "people" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["people"])
                        elif active_domain == "attraction":
                            if "pounds" in tokens[index+1] and "entrance fee" in " ".join(tokens[index-3:index]):
                                value = top_results[active_domain][result_idx]['entrance fee'].split()[
                                    0]
                                if "?" not in value:
                                    try:
                                        value = str(int(value))
                                    except:
                                        value = 'free'
                                else:
                                    value = "unknown"
                                response.append(value)
                        # if "there are" in " ".join(tokens[index-2:index]):
                            # response.append(str(num_results))
                        # elif "the" in tokens[index-2]: # the first [value_count], the last [value_count]
                            # response.append("1")
                        else:
                            response.append(str(num_results))
                    elif slot == 'place':
                        if 'arriv' in " ".join(tokens[index-2:index]) or "to" in " ".join(tokens[index-2:index]):
                            response.append(
                                top_results[active_domain][result_idx]["destination"])
                        elif 'leav' in " ".join(tokens[index-2:index]) or "from" in tokens[index-2:index] in "depart" in " ".join(tokens[index-2:index]):
                            response.append(
                                top_results[active_domain][result_idx]["departure"])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['destination', 'departure']:
                                        if s in state[d]:
                                            response.append(
                                                state[d][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'time':
                        if 'arriv' in ' '.join(response[-7:]) or 'arriving' in ' '.join(response[-7:]):
                            if active_domain is not None and 'arriveBy' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(
                                    top_results[active_domain][result_idx]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arriveBy' in state[d]:
                                    response.append(
                                        state[d]['arrive by'])
                                    break
                        elif 'leav' in ' '.join(response[-7:]) or 'depart' in ' '.join(response[-7:]):
                            if active_domain is not None and 'leaveAt' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(
                                    top_results[active_domain][result_idx]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leave at' in state[d]:
                                    response.append(
                                        state[d]['leave at'])
                                    break
                        elif 'book' in response or "booked" in response:
                            if state['restaurant']['book time'] != "":
                                response.append(
                                    state['restaurant']['book time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arrive by', 'leave at']:
                                        if s in state[d]:
                                            response.append(
                                                state[d][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price':
                        if active_domain == 'attraction':
                            value = top_results['attraction'][result_idx]['entrance fee'].split()[
                                0]
                            if "?" not in value:
                                try:
                                    value = str(int(value))
                                except:
                                    value = 'free'
                            else:
                                value = "unknown"
                            response.append(value)
                        elif active_domain == "train":
                            response.append(
                                top_results[active_domain][result_idx][slot].split()[0])
                    elif slot == "day" and active_domain in ["restaurant", "hotel"]:
                        if state[active_domain]['book day'] != "":
                            response.append(
                                state[active_domain]['book day'])

                    else:
                        # slot-filling based on query results
                        for d in top_results:
                            if slot in top_results[d][result_idx]:
                                response.append(
                                    top_results[d][result_idx][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]:
                                    response.append(state[d][slot])
                                    break
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'phone':
                            response.append('01223216297')
                        elif slot == 'department':
                            response.append('neurosciences critical care unit')
                        elif slot == 'address':
                            response.append("56 Lincoln street")
                        elif slot == "postcode":
                            response.append('533421')
                    elif domain == 'police':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'name':
                            response.append('Parkside Police Station')
                        elif slot == 'address':
                            response.append('Parkside, Cambridge')
                        elif slot == 'postcode':
                            response.append('533420')
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'color':
                            response.append('white')
                        elif slot == 'type':
                            response.append('toyota')
                    else:
                        # print(token)
                        response.append(token)
            else:
                response.append(token)

        try:
            response = ' '.join(response)
        except Exception as e:
            # pprint(response)
            raise
        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')
        # print(template, response)

        return response

    def model_predict(self, data_feed):
        self.logprobs = []
        logprobs, pred_labels, joint_logpz, sample_y = self.model.forward_rl(
            data_feed, self.model.config.max_dec_len)

        self.logprobs.extend(joint_logpz)

        pred_labels = np.array(
            [pred_labels], dtype=int)
        de_tknize = get_detokenize()
        pred_str = get_sent(self.model.vocab, de_tknize, pred_labels, 0)

        return pred_str

    def update(self, reward, logprobs):
        torch.autograd.set_detect_anomaly(True)
        reward = th.sum(reward).view(1, 1)
        if self.all_rewards == []:
            self.all_rewards = reward
        else:
            self.all_rewards = th.cat([self.all_rewards, reward])
        # standardize the reward
        r = (reward - th.mean(self.all_rewards)) / max(1e-4, th.std(self.all_rewards))
        # compute accumulated discounted reward
        # g = self.np2var(np.array([r]), FLOAT).view(1, 1)
        g = r
        rewards = []
        for _ in logprobs:
            rewards.insert(0, g)
            g = g * self.config.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, re in zip(logprobs, rewards):
            loss -= lp * re
        self.opt.zero_grad()
        if "fp16" in self.config and self.config.fp16:
            with amp.scale_loss(loss, self.opt) as scaled_loss:
               scaled_loss.backward() 
            nn.utils.clip_grad_norm_(amp.master_params(self.opt), self.config.grad_clip)
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.opt.step()
        
    def _print_grad(self):
        for name, p in self.model.named_parameters():
           print(name)
           print(p.grad)

    def save(self, postfix=None):
        if postfix is not None:
            th.save(self.model.state_dict(), self.rl_config.US_best_reward_model_path.replace("US", postfix))
        else:
            th.save(self.model.state_dict(), self.rl_config.US_best_reward_model_path)

def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)

def pad_to(max_len, tokens, do_pad):
    if len(tokens) >= max_len:
        return tokens[: max_len-1] + [tokens[-1]]
    elif do_pad:
        return tokens + [0] * (max_len - len(tokens))
    else:
        return tokens

def prepare_batch_gen(rows, config, pad_context=True):
    domains = ['hotel', 'restaurant', 'train',
               'attraction', 'hospital', 'police', 'taxi']

    ctx_utts, ctx_lens = [], []
    out_utts, out_lens = [], []

    out_bs, out_db = [], []
    goals, goal_lens = [], [[] for _ in range(len(domains))]
    keys = []

    for row in rows:
        in_row, out_row = row['context'], row['response']
        # source context
        batch_ctx = []
        for turn in in_row:
            batch_ctx.append(
                pad_to(config.max_utt_len, turn['utt'], do_pad=pad_context))
        ctx_utts.append(batch_ctx)
        ctx_lens.append(len(batch_ctx))

        out_bs.append(out_row['bs'])
        out_db.append(out_row['db'])

    batch_size = len(ctx_lens)
    vec_ctx_lens = np.array(ctx_lens)  # (batch_size, ), number of turns
    max_ctx_len = np.max(vec_ctx_lens)
    if pad_context:
        vec_ctx_utts = np.zeros(
            (batch_size, max_ctx_len, config.max_utt_len), dtype=np.int32)
    else:
        vec_ctx_utts = []
    vec_out_bs = np.array(out_bs)  # (batch_size, 94)
    vec_out_db = np.array(out_db)  # (batch_size, 30)

    for b_id in range(batch_size):
        if pad_context:
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
        else:
            vec_ctx_utts.append(ctx_utts[b_id])


    return Pack(context_lens=vec_ctx_lens,  # (batch_size, )
                # (batch_size, max_ctx_len, max_utt_len)
                contexts=vec_ctx_utts,
                bs=vec_out_bs,  # (batch_size, 94)
                db=vec_out_db  # (batch_size, 30)
                )


if __name__ == '__main__':

    domain_name = 'object_division'
    domain_info = domain.get_domain(domain_name)

    train_data_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'data/norm-multi-woz/train_dials.json')

    config = Pack(
        seed=10,
        train_path=train_data_path,
        max_vocab_size=1000,
        last_n_model=5,
        max_utt_len=50,
        max_dec_len=50,
        backward_size=2,
        batch_size=1,
        use_gpu=True,
        op='adam',
        init_lr=0.001,
        l2_norm=1e-05,
        momentum=0.0,
        grad_clip=5.0,
        dropout=0.5,
        max_epoch=100,
        embed_size=100,
        num_layers=1,
        utt_rnn_cell='gru',
        utt_cell_size=300,
        bi_utt_cell=True,
        enc_use_attn=True,
        dec_use_attn=True,
        dec_rnn_cell='lstm',
        dec_cell_size=300,
        dec_attn_mode='cat',
        y_size=10,
        k_size=20,
        beta=0.001,
        simple_posterior=True,
        contextual_posterior=True,
        use_mi=False,
        use_pr=True,
        use_diversity=False,
        #
        beam_size=20,
        fix_batch=True,
        fix_train_batch=False,
        avg_type='word',
        print_step=300,
        ckpt_step=1416,
        improve_threshold=0.996,
        patient_increase=2.0,
        save_model=True,
        early_stop=False,
        gen_type='greedy',
        preview_batch_num=None,
        k=domain_info.input_length(),
        init_range=0.1,
        pretrain_folder='2019-09-20-21-43-06-sl_cat',
        forward_only=False
    )

    state = {'user_action': [["Inform", "Hotel", "Area", "east"], ["Inform", "Hotel", "Stars", "4"]],
             'system_action': [],
             'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                              'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                        'semi': {'name': '',
                                                 'area': 'east',
                                                 'parking': '',
                                                 'pricerange': '',
                                                 'stars': '4',
                                                 'internet': '',
                                                 'type': ''}},
                              'attraction': {'book': {'booked': []},
                                             'semi': {'type': '', 'name': '', 'area': ''}},
                              'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                             'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                              'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                              'taxi': {'book': {'booked': []},
                                       'semi': {'leaveAt': '',
                                                'destination': '',
                                                'departure': '',
                                                'arriveBy': ''}},
                              'train': {'book': {'booked': [], 'people': ''},
                                        'semi': {'leaveAt': '',
                                                 'destination': '',
                                                 'day': '',
                                                 'arriveBy': '',
                                                 'departure': ''}}},
             'request_state': {},
             'terminated': False,
             'history': [['sys', ''],
                         ['user', 'Could you book a 4 stars hotel east of town for one night, 1 person?']]}

    model_file="path/to/model" # points to model from lava repo
    cur_model = LAVA(model_file)

    response = cur_model.predict(state)
    # print(response)
