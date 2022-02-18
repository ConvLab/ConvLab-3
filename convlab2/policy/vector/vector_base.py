# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import copy
import logging
from convlab2.policy.vec import Vector
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

DEFAULT_INTENT_FILEPATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    'data/multiwoz/trackable_intent.json'
)

root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)


SLOT_MAP = {'taxi_types': 'car type'}


class MultiWozVectorBase(Vector):

    def __init__(self, voc_file=None, voc_opp_file=None,
                 intent_file=DEFAULT_INTENT_FILEPATH, character='sys',
                 use_masking=False,
                 manually_add_entity_names=True,
                 seed=0):

        super().__init__()

        self.set_seed(seed)
        self.belief_domains = ['Attraction', 'Restaurant',
                               'Train', 'Hotel', 'Taxi', 'Hospital', 'Police']
        self.db_domains = ['Attraction', 'Restaurant', 'Train', 'Hotel']
        self.max_actionval = {}

        with open(intent_file) as f:
            intents = json.load(f)
        self.informable = intents['informable']
        self.requestable = intents['requestable']
        self.db = Database()

        self.use_mask = use_masking
        self.use_add_name = manually_add_entity_names
        self.reqinfo_filler_action = None
        self.character = character

        self.name_history_flag = True
        self.name_action_prev = []

        if not voc_file or not voc_opp_file:
            voc_file = os.path.join(
                root_dir, 'data/multiwoz/sys_da_voc_remapped.txt')
            voc_opp_file = os.path.join(
                root_dir, 'data/multiwoz/usr_da_voc.txt')

        with open(voc_file) as f:
            self.da_voc = f.read().splitlines()
        with open(voc_opp_file) as f:
            self.da_voc_opp = f.read().splitlines()

        self.generate_dict()
        self.cur_domain = None
        self.get_state_dim()
        self.state = default_state()

    def get_state_dim(self):
        '''
        Compute the state dimension for the policy input
        '''
        self.state_dim = 0
        raise NotImplementedError

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (tuple):
                Dialog state
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        raise NotImplementedError

    def set_seed(self, seed):
        np.random.seed(seed)

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)
        self.opp2vec = dict((a, i) for i, a in enumerate(self.da_voc_opp))
        self.da_opp_dim = len(self.da_voc_opp)

    def retrieve_user_action(self, state):

        action = state['user_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                opp_act_vec[self.opp2vec[da]] = 1.
        return opp_act_vec

    def compute_domain_mask(self, domain_active_dict):

        mask_list = np.zeros(self.da_dim)

        for i in range(self.da_dim):
            action = self.vec2act[i]
            action_domain = action.split('-')[0]
            if action_domain in domain_active_dict.keys():
                if not domain_active_dict[action_domain]:
                    mask_list[i] = 1.0

        return mask_list

    def compute_general_mask(self):

        mask_list = np.zeros(self.da_dim)

        for i in range(self.da_dim):
            action = self.vec2act[i]
            domain, intent, slot, value = action.split('-')

            # NoBook-SLOT does not make sense because policy can not know which constraint made booking impossible
            # If one wants to do it, lexicaliser needs to do it
            if intent.lower() in ['nobook', 'nooffer'] and slot.lower() != 'none':
                mask_list[i] = 1.0

            # see policy/rule/multiwoz/policy_agenda_multiwoz.py: illegal booking slot. Is self.cur_domain correct?
            if self.cur_domain is not None:
                if slot.lower() == 'time' and self.cur_domain.lower() not in ['train', 'restaurant']:
                    if domain.lower() == 'booking':
                        mask_list[i] = 1.0

                if slot.lower() in self.state[self.cur_domain.lower()]['book']:
                    if not self.state[self.cur_domain.lower()]['book'][slot.lower()] and intent.lower() == 'inform':
                        mask_list[i] = 1.0

            if domain.lower() == 'taxi':
                slot = REF_SYS_DA.get(domain, {}).get(slot, slot.lower())
                if slot in self.state['taxi']['semi']:
                    if not self.state['taxi']['semi'][slot] and intent.lower() == 'inform':
                        mask_list[i] = 1.0

        return mask_list

    def compute_entity_mask(self, number_entities_dict):
        mask_list = np.zeros(self.da_dim)
        for i in range(self.da_dim):
            action = self.vec2act[i]
            domain, intent, slot, value = action.split('-')
            domain_entities = number_entities_dict.get(domain, 1)

            if intent.lower() in ['inform', 'select', 'recommend'] and value != None and value != 'none':
                if(int(value) > domain_entities):
                    mask_list[i] = 1.0

            if intent.lower() in ['inform', 'select', 'recommend'] and domain.lower() in ['booking']:
                if number_entities_dict.get(self.cur_domain, 0) == 0:
                    mask_list[i] = 1.0

            # mask Booking-NoBook if an entity is available in the current domain
            if intent.lower() in ['nobook'] and number_entities_dict.get(self.cur_domain, 0) > 0:
                mask_list[i] = 1.0

            if intent.lower() in ['nooffer'] and number_entities_dict.get(domain, 0) > 0:
                mask_list[i] = 1.0

        return mask_list

    def dbquery_domain(self, domain):
        """
        query entities of specified domain
        Args:
            domain string:
                domain to query
        Returns:
            entities list:
                list of entities of the specified domain
        """
        constraint = self.state[domain.lower()]['semi']
        constraint = {k: i for k, i in constraint.items() if i and i not in [
            'dontcare', "do n't care", "do not care"]}

        return self.db.query(domain.lower(), constraint.items())

    # Function used to find which user constraint results in no entities being found

    def find_nooffer_slot(self, domain):
        """
        query entities of specified domain
        Args:
            domain string:
                domain to query
        Returns:
            entities list:
                list of entities of the specified domain
        """
        constraint = self.state[domain.lower()]['semi']
        constraint = {k: i for k, i in constraint.items() if i and i not in [
            'dontcare', "do n't care", "do not care"]}

        # Leave slots out of constraints to find which slot constraint results in no entities being found
        for slot in constraint:
            constraint_ = {k: i for k, i in constraint.items()
                           if k != slot}.items()
            entities = self.db.query(domain.lower(), constraint_)
            if entities:
                return slot

        # If no single slot results in no entities being found try the above with pairs of slots
        slots = [slot for slot in constraint]
        pairs = []
        for i, slot in enumerate(slots):
            for j, slot1 in enumerate(slots):
                if j > i:
                    pairs.append((slot, slot1))

        for slot, slot1 in pairs:
            constraint_ = {k: i for k, i in constraint.items(
            ) if k != slot and k != slot1}.items()
            entities = self.db.query(domain.lower(), constraint_)
            if entities:
                return np.random.choice([slot, slot1])

        # If no single slots or pairs removed results in success then set slot 'none'
        return 'none'

    def action_vectorize(self, action):
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        act_vec = np.zeros(self.da_dim)

        for da in action:
            if da in self.act2vec:
                act_vec[self.act2vec[da]] = 1.
        return act_vec

    def action_devectorize(self, action_vec):
        """
        recover an action
        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple):
                Dialog act
        """

        act_array = []

        for i, idx in enumerate(action_vec):
            if idx == 1:
                act_array.append(self.vec2act[i])

        if len(act_array) == 0:
            if self.reqinfo_filler_action:
                act_array.append('general-reqinfo-none-none')
            else:
                act_array.append('general-reqmore-none-none')

        action = deflat_da(act_array)
        entities = {}
        for domint in action:
            domain, intent = domint.split('-')
            if domain not in entities and domain.lower() not in ['general', 'booking']:
                entities[domain] = self.dbquery_domain(domain)
        if self.cur_domain and self.cur_domain not in entities:
            entities[self.cur_domain] = self.dbquery_domain(self.cur_domain)

        nooffer = [domint for domint in action if 'NoOffer' in domint]
        for domint in nooffer:
            domain, intent = domint.split('-')
            slot = self.find_nooffer_slot(domain)
            slot = REF_USR_DA.get(domain, {slot: 'none'}).get(slot, 'none')
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        nobook = [domint for domint in action if 'NoBook' in domint]
        for domint in nobook:
            domain = self.cur_domain if self.cur_domain else 'none'
            if domain.lower() in self.state:
                slots = self.state[domain.lower()]['book']
                slots = [slot for slot, i in slots.items()
                         if i and slot != 'booked']
                slots.append('none')
                slot = np.random.choice(slots)
                slot = REF_USR_DA.get(domain, {}).get(slot, 'none')
            else:
                slot = 'none'
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        # When there is a INFORM(1 name) or OFFER(multiple) action then inform the name

        if self.use_add_name:
            action = self.add_name(action)

        for key in action.keys():
            index = -1
            for [item, idx] in action[key]:
                if index != -1 and index != idx and idx != '?':
                    logging.debug(
                        "System is likely refering multiple entities within this turn")
                    logging.debug(action[key])
                index = idx

        action = lexicalize_da(action, entities, self.state,
                               self.requestable, self.cur_domain)

        return action

    def add_name(self, action):

        name_inform = []
        contains_name = False
        # General Inform Condition for Naming
        cur_inform = str(self.cur_domain) + '-Inform'
        cur_request = str(self.cur_domain) + '-Request'
        index = -1
        if cur_inform in action:
            for [item, idx] in action[cur_inform]:
                if item == 'Name':
                    contains_name = True
                elif self.cur_domain == 'Train' and item == 'Id':
                    contains_name = True
                elif self.cur_domain == 'Hospital':
                    contains_name = True
                elif item == 'Choice' and cur_request in action:
                    contains_name = True

                if index != -1 and index != idx and idx is not None:
                    logging.debug(
                        "System is likely refering multiple entities within this turn")

                index = idx

            if contains_name == False:
                if self.cur_domain == 'Train':
                    name_act = ['Id', index]
                else:
                    name_act = ['Name', index]

                tmp = [name_act] + action[cur_inform]
                name_inform = name_act

                if self.name_history_flag:
                    action[cur_inform] = tmp

        if self.name_action_prev != []:
            if name_inform == self.name_action_prev:
                self.name_history_flag = False
            else:
                self.name_history_flag = True

        if name_inform != []:
            self.name_action_prev = copy.deepcopy(name_inform)

        return action
