# -*- coding: utf-8 -*-
import sys
import numpy as np
from convlab.util.multiwoz.lexicalize import delexicalize_da, flat_da
from .vector_base import VectorBase


class VectorBinary(VectorBase):

    def __init__(self, dataset_name='multiwoz21', character='sys', use_masking=False, manually_add_entity_names=True,
                 seed=0, **kwargs):

        super().__init__(dataset_name, character, use_masking, manually_add_entity_names, seed)

    def get_state_dim(self):
        self.belief_state_dim = 0

        for domain in self.ontology['state']:
            for slot in self.ontology['state'][domain]:
                self.belief_state_dim += 1

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
            len(self.db_domains) + 6 * len(self.db_domains) + 1

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
        domain_active_dict = self.init_domain_active_dict()

        # when character is sys, to help query database when da is booking-book
        # update current domain according to user action
        if self.character == 'sys':
            action = state['user_action']
            for intent, domain, slot, value in action:
                domain_active_dict[domain] = True

        opp_act_vec = self.vectorize_user_act(state)
        last_act_vec = self.vectorize_system_act(state)
        belief_state, domain_active_dict = self.vectorize_belief_state(state, domain_active_dict)
        book = self.vectorize_booked(state)
        degree, number_entities_dict = self.pointer()
        final = 1. if state['terminated'] else 0.

        state_vec = np.r_[opp_act_vec, last_act_vec,
                          belief_state, book, degree, final]
        assert len(state_vec) == self.state_dim

        if self.use_mask:
            mask = self.get_mask(domain_active_dict, number_entities_dict)
            for i in range(self.da_dim):
                mask[i] = -int(bool(mask[i])) * sys.maxsize
        else:
            mask = np.zeros(self.da_dim)

        return state_vec, mask

    def get_mask(self, domain_active_dict, number_entities_dict):
        #domain_mask = self.compute_domain_mask(domain_active_dict)
        entity_mask = self.compute_entity_mask(number_entities_dict)
        general_mask = self.compute_general_mask()
        mask = entity_mask + general_mask
        return mask

    def vectorize_booked(self, state):
        book = np.zeros(len(self.db_domains))
        for i, domain in enumerate(self.db_domains):
            if domain in state['booked'] and state['booked'][domain]:
                book[i] = 1.
        return book

    def vectorize_belief_state(self, state, domain_active_dict):
        belief_state = np.zeros(self.belief_state_dim)
        i = 0
        for domain in self.belief_domains:
            for slot, value in state['belief_state'][domain].items():
                if value:
                    belief_state[i] = 1.
                i += 1

            if [slot for slot, value in state['belief_state'][domain].items() if value]:
                domain_active_dict[domain] = True
        return belief_state, domain_active_dict

    def vectorize_system_act(self, state):
        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        last_act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                last_act_vec[self.act2vec[da]] = 1.
        return last_act_vec

    def vectorize_user_act(self, state):
        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                prob = 1.0
                opp_act_vec[self.opp2vec[da]] = prob
        return opp_act_vec
