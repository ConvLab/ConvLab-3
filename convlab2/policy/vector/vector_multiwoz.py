# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da
from convlab2.util.multiwoz.state import default_state
from .vector_base import MultiWozVectorBase

DEFAULT_INTENT_FILEPATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    'data/multiwoz/trackable_intent.json'
)


SLOT_MAP = {'taxi_types': 'car type'}


class MultiWozVector(MultiWozVectorBase):

    def __init__(self, voc_file=None, voc_opp_file=None, character='sys',
                 intent_file=DEFAULT_INTENT_FILEPATH,
                 use_masking=False,
                 manually_add_entity_names=True,
                 seed=0):

        super().__init__(voc_file, voc_opp_file, intent_file, character, use_masking, manually_add_entity_names, seed)

    def get_state_dim(self):
        self.belief_state_dim = 0
        for domain in self.belief_domains:
            for slot, value in default_state()['belief_state'][domain.lower()]['semi'].items():
                self.belief_state_dim += 1

            self.belief_state_dim += len(default_state()['belief_state'][domain.lower()]['book']) - 1

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
            len(self.db_domains) + 6 * len(self.db_domains) + 1

    def pointer(self):
        pointer_vector = np.zeros(6 * len(self.db_domains))
        number_entities_dict = {}
        for domain in self.db_domains:
            entities = self.dbquery_domain(domain.lower())
            number_entities_dict[domain] = len(entities)
            pointer_vector = self.one_hot_vector(
                len(entities), domain, pointer_vector)

        return pointer_vector, number_entities_dict

    def one_hot_vector(self, num, domain, vector):
        """Return number of available entities for particular domain."""
        if domain != 'train':
            idx = self.db_domains.index(domain)
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
            idx = self.db_domains.index(domain)
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

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (dict):
                Dialog state
            action (tuple):
                Dialog act
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        self.state = state['belief_state']
        self.confidence_scores = state['belief_state_probs'] if 'belief_state_probs' in state else None
        domain_active_dict = {}
        for domain in self.belief_domains:
            domain_active_dict[domain] = False

        # when character is sys, to help query database when da is booking-book
        # update current domain according to user action
        if self.character == 'sys':
            action = state['user_action']
            for intent, domain, slot, value in action:
                domain_active_dict[domain] = True
                if domain in self.db_domains:
                    self.cur_domain = domain

        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)

        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                prob = 1.0
                opp_act_vec[self.opp2vec[da]] = prob

        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        last_act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                last_act_vec[self.act2vec[da]] = 1.

        belief_state = np.zeros(self.belief_state_dim)
        i = 0
        for domain in self.belief_domains:

            for slot, value in state['belief_state'][domain.lower()]['semi'].items():
                if value and value != 'not mentioned':
                    belief_state[i] = 1.
                i += 1
            for slot, value in state['belief_state'][domain.lower()]['book'].items():
                if slot == 'booked':
                    continue
                if value and value != "not mentioned":
                    belief_state[i] = 1.

            if 'active_domains' in state:
                domain_active = state['active_domains'][domain.lower()]
                domain_active_dict[domain] = domain_active
                if domain in self.db_domains and domain_active:
                    self.cur_domain = domain
            else:
                if [slot for slot, value in state['belief_state'][domain.lower()]['semi'].items() if value]:
                    domain_active_dict[domain] = True

        book = np.zeros(len(self.db_domains))
        for i, domain in enumerate(self.db_domains):
            if state['belief_state'][domain.lower()]['book']['booked']:
                book[i] = 1.

        degree, number_entities_dict = self.pointer()

        final = 1. if state['terminated'] else 0.

        state_vec = np.r_[opp_act_vec, last_act_vec,
                          belief_state, book, degree, final]
        assert len(state_vec) == self.state_dim

        if self.use_mask is not None:
            # None covers the case for policies that don't use masking at all, so do not expect an output "state_vec, mask"
            if self.use_mask:
                domain_mask = self.compute_domain_mask(domain_active_dict)
                entity_mask = self.compute_entity_mask(number_entities_dict)
                general_mask = self.compute_general_mask()
                mask = domain_mask + entity_mask + general_mask
                for i in range(self.da_dim):
                    mask[i] = -int(bool(mask[i])) * sys.maxsize
            else:
                mask = np.zeros(self.da_dim)

            return state_vec, mask
        else:
            return state_vec
