# -*- coding: utf-8 -*-
import sys
import numpy as np
import logging

from convlab.util.multiwoz.lexicalize import delexicalize_da, flat_da
from .vector_base import VectorBase


class VectorNodes(VectorBase):

    def __init__(self, dataset_name='multiwoz21', character='sys', use_masking=False, manually_add_entity_names=True,
                 seed=0, filter_state=True):

        super().__init__(dataset_name, character, use_masking, manually_add_entity_names, seed)
        self.filter_state = filter_state
        logging.info(f"We filter state by active domains: {self.filter_state}")

    def get_state_dim(self):
        self.belief_state_dim = 0

        for domain in self.ontology['state']:
            for slot in self.ontology['state'][domain]:
                self.belief_state_dim += 1

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
            len(self.db_domains) + 6 * len(self.db_domains) + 1

    def init_kg_graph(self):
        self.kg_info = []

    def add_graph_node(self, domain, node_type, description, value):

        node = {"domain": domain, "node_type": node_type, "description": description, "value": value}
        self.kg_info.append(node)

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
        self.init_kg_graph()

        # when character is sys, to help query database when da is booking-book
        # update current domain according to user action
        if self.character == 'sys':
            action = state['user_action']
            for intent, domain, slot, value in action:
                domain_active_dict[domain] = True

        self.get_user_act_feature(state)
        self.get_sys_act_feature(state)
        domain_active_dict = self.get_user_goal_feature(state, domain_active_dict)
        self.get_general_features(state, domain_active_dict)

        if self.db is not None:
            number_entities_dict = self.get_db_features()
        else:
            number_entities_dict = None

        if self.filter_state:
            self.kg_info = self.filter_inactive_domains(domain_active_dict)

        if self.use_mask:
            mask = self.get_mask(domain_active_dict, number_entities_dict)
            for i in range(self.da_dim):
                mask[i] = -int(bool(mask[i])) * sys.maxsize
        else:
            mask = np.zeros(self.da_dim)

        return np.zeros(1), mask

    def get_mask(self, domain_active_dict, number_entities_dict):
        #domain_mask = self.compute_domain_mask(domain_active_dict)
        entity_mask = self.compute_entity_mask(number_entities_dict)
        general_mask = self.compute_general_mask()
        mask = entity_mask + general_mask
        return mask

    def get_db_features(self):

        degree, number_entities_dict = self.pointer()
        feature_type = 'db'
        for domain, num_entities in number_entities_dict.items():
            description = f"db-{domain}-entities".lower()
            # self.add_graph_node(domain, feature_type, description, int(num_entities > 0))
            self.add_graph_node(domain, feature_type, description, min(num_entities, 5) / 5)
        return number_entities_dict

    def get_user_goal_feature(self, state, domain_active_dict):

        feature_type = 'user goal'
        for domain in self.belief_domains:
            # the if case is needed because SGD only saves the dialogue state info for active domains
            if domain in state['belief_state']:
                for slot, value in state['belief_state'][domain].items():
                    description = f"user goal-{domain}-{slot}".lower()
                    value = 1.0 if (value and value != "not mentioned") else 0.0
                    self.add_graph_node(domain, feature_type, description, value)

                if [slot for slot, value in state['belief_state'][domain].items() if value]:
                    domain_active_dict[domain] = True
        return domain_active_dict

    def get_sys_act_feature(self, state):

        feature_type = 'last system act'
        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        for da in action:
            if da in self.act2vec:
                domain = da.split('-')[0]
                description = "system-" + da
                value = 1.0
                self.add_graph_node(domain, feature_type, description.lower(), value)

    def get_user_act_feature(self, state):
        # user-act feature
        feature_type = 'user act'
        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)

        for da in opp_action:
            if da in self.opp2vec:
                domain = da.split('-')[0]
                description = "user-" + da
                value = 1.0
                self.add_graph_node(domain, feature_type, description.lower(), value)

    def get_general_features(self, state, domain_active_dict):

        feature_type = 'general'
        if 'booked' in state:
            for i, domain in enumerate(self.db_domains):
                if domain in state['booked']:
                    description = f"general-{domain}-booked".lower()
                    value = 1.0 if state['booked'][domain] else 0.0
                    self.add_graph_node(domain, feature_type, description, value)

        for domain in self.domains:
            if domain == 'general':
                continue
            value = 1.0 if domain_active_dict[domain] else 0
            description = f"general-{domain}".lower()
            self.add_graph_node(domain, feature_type, description, value)

    def filter_inactive_domains(self, domain_active_dict):

        kg_filtered = []
        for node in self.kg_info:
            domain = node['domain']
            if domain in domain_active_dict:
                if domain_active_dict[domain]:
                    kg_filtered.append(node)
            else:
                kg_filtered.append(node)

        return kg_filtered

