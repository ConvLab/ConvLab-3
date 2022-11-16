# -*- coding: utf-8 -*-
import sys
import numpy as np
import logging

from convlab.util.multiwoz.lexicalize import delexicalize_da, flat_da
from convlab.policy.vector.vector_binary import VectorBinary


class VectorUncertainty(VectorBinary):
    """Vectorise state and state uncertainty predictions"""

    def __init__(self,
                 dataset_name: str = 'multiwoz21',
                 character: str = 'sys',
                 use_masking: bool = False,
                 manually_add_entity_names: bool = True,
                 seed: str = 0,
                 use_confidence_scores: bool = True,
                 confidence_thresholds: dict = None,
                 use_state_total_uncertainty: bool = False,
                 use_state_knowledge_uncertainty: bool = False):
        """
        Args:
            dataset_name: Name of environment dataset
            character: Character of the agent (sys/usr)
            use_masking: If true certain actions are masked during devectorisation
            manually_add_entity_names: If true inform entity name actions are manually added
            seed: Seed
            use_confidence_scores: If true confidence scores are used in state vectorisation
            confidence_thresholds: If true confidence thresholds are used in database querying
            use_state_total_uncertainty: If true state entropy is added to the state vector
            use_state_knowledge_uncertainty: If true state mutual information is added to the state vector
        """

        self.use_confidence_scores = use_confidence_scores
        self.use_state_total_uncertainty = use_state_total_uncertainty
        self.use_state_knowledge_uncertainty = use_state_knowledge_uncertainty
        if confidence_thresholds is not None:
            self.setup_uncertain_query(confidence_thresholds)

        super().__init__(dataset_name, character, use_masking, manually_add_entity_names, seed)

    def get_state_dim(self):
        self.belief_state_dim = 0

        for domain in self.ontology['state']:
            for slot in self.ontology['state'][domain]:
                # Dim 1 - indicator/confidence score
                # Dim 2 - Entropy (Total uncertainty) / Mutual information (knowledge unc)
                slot_dim = 1 if not self.use_state_total_uncertainty else 2
                slot_dim += 1 if self.use_state_knowledge_uncertainty else 0
                self.belief_state_dim += slot_dim

        self.state_dim = self.da_opp_dim + self.da_dim + self.belief_state_dim + \
            len(self.db_domains) + 6 * len(self.db_domains) + 1

    # Add thresholds for db_queries
    def setup_uncertain_query(self, confidence_thresholds):
        self.use_confidence_scores = True
        self.confidence_thresholds = confidence_thresholds
        logging.info('DB Search uncertainty activated.')

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
        constraints = {slot: value for slot, value in self.state[domain].items()
                       if slot and value not in ['dontcare',
                                                 "do n't care", "do not care"]} if domain in self.state else dict()

        # Remove constraints for which the uncertainty is high
        if self.confidence_scores is not None and self.use_confidence_scores and self.confidence_thresholds is not None:
            # Collect threshold values for each domain-slot pair
            threshold = self.confidence_thresholds.get(domain, dict())
            threshold = {slot: threshold.get(slot, 0.05) for slot in constraints}
            # Get confidence scores for each constraint
            probs = self.confidence_scores.get(domain, dict())
            probs = {slot: probs.get(slot, {}).get('inform', 1.0) for slot in constraints}

            # Filter out constraints for which confidence is lower than threshold
            constraints = {slot: value for slot, value in constraints.items() if probs[slot] >= threshold[slot]}

        return self.db.query(domain, constraints.items(), topk=10)

    def vectorize_user_act(self, state):
        """Return confidence scores for the user actions"""
        self.confidence_scores = state['belief_state_probs'] if 'belief_state_probs' in state else None
        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                if 'belief_state_probs' in state and self.use_confidence_scores:
                    domain, intent, slot, value = da.split('-')
                    if domain in state['belief_state_probs']:
                        slot = slot if slot else 'none'
                        if slot in state['belief_state_probs'][domain]:
                            prob = state['belief_state_probs'][domain][slot]
                        elif slot.lower() in state['belief_state_probs'][domain]:
                            prob = state['belief_state_probs'][domain][slot.lower()]
                        else:
                            prob = dict()

                        if intent in prob:
                            prob = float(prob[intent])
                        else:
                            prob = 1.0
                    else:
                        prob = 1.0
                else:
                    prob = 1.0
                opp_act_vec[self.opp2vec[da]] = prob

        return opp_act_vec

    def vectorize_belief_state(self, state, domain_active_dict):
        belief_state = np.zeros(self.belief_state_dim)
        i = 0
        for domain in self.belief_domains:
            if self.use_confidence_scores and 'belief_state_probs' in state:
                for slot in state['belief_state'][domain]:
                    prob = None
                    if slot in state['belief_state_probs'][domain]:
                        prob = state['belief_state_probs'][domain][slot]
                        prob = prob['inform'] if 'inform' in prob else None
                    if prob:
                        belief_state[i] = float(prob)
                    i += 1
            else:
                for slot, value in state['belief_state'][domain].items():
                    if value and value != 'not mentioned':
                        belief_state[i] = 1.
                    i += 1

            if 'active_domains' in state:
                domain_active = state['active_domains'][domain]
                domain_active_dict[domain] = domain_active
            else:
                if [slot for slot, value in state['belief_state'][domain].items() if value]:
                    domain_active_dict[domain] = True

        # Add knowledge and/or total uncertainty to the belief state
        if self.use_state_total_uncertainty and 'entropy' in state:
            for domain in self.belief_domains:
                for slot in state['belief_state'][domain]:
                    if slot in state['entropy'][domain]:
                        belief_state[i] = float(state['entropy'][domain][slot])
                    i += 1

        if self.use_state_knowledge_uncertainty and 'mutual_information' in state:
            for domain in self.belief_domains:
                for slot in state['belief_state'][domain]:
                    if slot in state['mutual_information'][domain]:
                        belief_state[i] = float(state['mutual_information'][domain][slot])
                    i += 1

        return belief_state, domain_active_dict
