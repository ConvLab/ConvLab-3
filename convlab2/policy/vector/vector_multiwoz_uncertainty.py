# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import logging
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from .vector_binary import VectorBinary as VectorBase

DEFAULT_INTENT_FILEPATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    'data/multiwoz/trackable_intent.json'
)


SLOT_MAP = {'taxi_types': 'car type'}


class MultiWozVector(VectorBase):

    def __init__(self, voc_file=None, voc_opp_file=None, character='sys',
                 intent_file=DEFAULT_INTENT_FILEPATH,
                 use_confidence_scores=False,
                 use_entropy=False,
                 use_mutual_info=False,
                 use_masking=False,
                 manually_add_entity_names=False,
                 seed=0,
                 shrink=False):

        self.use_confidence_scores = use_confidence_scores
        self.use_entropy = use_entropy
        self.use_mutual_info = use_mutual_info
        self.thresholds = None

        super().__init__(voc_file, voc_opp_file, character, intent_file, use_masking, manually_add_entity_names, seed)

    def get_state_dim(self):
        self.belief_state_dim = 0
        for domain in self.belief_domains:
            for slot in default_state()['belief_state'][domain.lower()]['semi']:
                # Dim 1 - indicator/confidence score
                # Dim 2 - Entropy (Total uncertainty) / Mutual information (knowledge unc)
                slot_dim = 1 if not self.use_entropy else 2
                slot_dim += 1 if self.use_mutual_info else 0
                self.belief_state_dim += slot_dim

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
            len(self.db_domains) + 6 * len(self.db_domains) + 1

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
        # Get all user constraints
        constraint = self.state[domain.lower()]['semi']
        constraint = {k: i for k, i in constraint.items() if i and i not in ['dontcare', "do n't care", "do not care"]}

        # Remove constraints for which the uncertainty is high
        if self.confidence_scores is not None and self.use_confidence_scores and self.thresholds != None:
            # Collect threshold values for each domain-slot pair
            thres = self.thresholds.get(domain.lower(), {})
            thres = {k: thres.get(k, 0.05) for k in constraint}
            # Get confidence scores for each constraint
            probs = self.confidence_scores.get(domain.lower(), {})
            probs = {k: probs.get(k, {}).get('inform', 1.0)
                     for k in constraint}

            # Filter out constraints for which confidence is lower than threshold
            constraint = {k: i for k, i in constraint.items()
                          if probs[k] >= thres[k]}

        return self.db.query(domain.lower(), constraint.items())

    # Add thresholds for db_queries
    def setup_uncertain_query(self, thresholds):
        self.use_confidence_scores = True
        self.thresholds = thresholds
        logging.info('DB Search uncertainty activated.')

    def vectorize_user_act_confidence_scores(self, state, opp_action):
        """Return confidence scores for the user actions"""
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in self.opp2vec:
            domain, intent, slot, value = da.split('-')
            if domain.lower() in state['belief_state_probs']:
                # Map slot name to match user actions
                slot = REF_SYS_DA[domain].get(
                    slot, slot) if domain in REF_SYS_DA else slot
                slot = slot if slot else 'none'
                slot = SLOT_MAP.get(slot, slot)
                domain = domain.lower()

                if slot in state['belief_state_probs'][domain]:
                    prob = state['belief_state_probs'][domain][slot]
                elif slot.lower() in state['belief_state_probs'][domain]:
                    prob = state['belief_state_probs'][domain][slot.lower()]
                else:
                    prob = {}

                intent = intent.lower()
                if intent in prob:
                    prob = float(prob[intent])
                elif da in opp_action:
                    prob = 1.0
                else:
                    prob = 0.0
            elif da in opp_action:
                prob = 1.0
            else:
                prob = 0.0
            opp_act_vec[self.opp2vec[da]] = prob

        return opp_act_vec

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

        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        if 'belief_state_probs' in state and self.use_confidence_scores:
            opp_act_vec = self.vectorize_user_act_confidence_scores(
                state, opp_action)
        else:
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
            if self.use_confidence_scores and 'belief_state_probs' in state:
                for slot in state['belief_state'][domain.lower()]['semi']:
                    if slot in state['belief_state_probs'][domain.lower()]:
                        prob = state['belief_state_probs'][domain.lower()
                                                           ][slot]
                        prob = prob['inform'] if 'inform' in prob else None
                    if prob:
                        belief_state[i] = float(prob)
                    i += 1
            else:
                for slot, value in state['belief_state'][domain.lower()]['semi'].items():
                    if value and value != 'not mentioned':
                        belief_state[i] = 1.
                    i += 1
            if 'active_domains' in state:
                domain_active = state['active_domains'][domain.lower()]
                domain_active_dict[domain] = domain_active
            else:
                if [slot for slot, value in state['belief_state'][domain.lower()]['semi'].items() if value]:
                    domain_active_dict[domain] = True

        # Add knowledge and/or total uncertainty to the belief state
        if self.use_entropy and 'entropy' in state:
            for domain in self.belief_domains:
                for slot in state['belief_state'][domain.lower()]['semi']:
                    if slot in state['entropy'][domain.lower()]:
                        belief_state[i] = float(
                            state['entropy'][domain.lower()][slot])
                    i += 1

        if self.use_mutual_info and 'mutual_information' in state:
            for domain in self.belief_domains:
                for slot in state['belief_state'][domain.lower()]['semi']:
                    if slot in state['mutual_information'][domain.lower()]:
                        belief_state[i] = float(
                            state['mutual_information'][domain.lower()][slot])
                    i += 1

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
