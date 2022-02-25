# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import copy
import logging
from copy import deepcopy
from convlab2.policy.vec import Vector
from convlab2.util.custom_util import flatten_acts
from convlab2.util.multiwoz.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

from convlab2.util import load_ontology, load_database, load_dataset

DEFAULT_INTENT_FILEPATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    'data/multiwoz/trackable_intent.json'
)

root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)


SLOT_MAP = {'taxi_types': 'car type'}

#TODO: The masks depend on multiwoz, deal with that somehow, shall we build a Mask class?
#TODO: Check the masks with new action strings
#TODO: Where should i save the action dicts?
#TODO: Load actions from ontology properly
#TODO: method AddName is properly not working right anymore


class VectorBase(Vector):

    def __init__(self, dataset_name='multiwoz21', character='sys', use_masking=False, manually_add_entity_names=True,
                 seed=0):

        super().__init__()

        self.set_seed(seed)
        self.ontology = load_ontology(dataset_name)
        try:
            self.db = load_database(dataset_name)
            self.db_domains = self.db.domains
        except:
            self.db = None
            self.db_domains = None
            print("VectorBase: Can not load a database, path is probably not existing.")

        self.dataset_name = dataset_name
        self.max_actionval = {}
        self.use_mask = use_masking
        self.use_add_name = manually_add_entity_names
        self.reqinfo_filler_action = None
        self.character = character
        self.name_history_flag = True
        self.name_action_prev = []
        self.cur_domain = None
        self.requestable = ['request']
        self.informable = ['inform', 'recommend']

        self.load_attributes()
        self.get_state_dim()
        print(f"State dimension: {self.state_dim}")

    def load_attributes(self):

        self.domains = list(self.ontology['domains'].keys())
        self.domains.sort()

        self.state = self.ontology['state']
        self.belief_domains = list(self.state.keys())
        self.belief_domains.sort()

        self.load_action_dicts()

    def load_action_dicts(self):

        self.load_actions_from_data()
        self.generate_dict()

    def load_actions_from_data(self, frequency_threshold=50):

        data_split = load_dataset(self.dataset_name)
        system_dict = {}
        user_dict = {}
        for key in data_split:
            data = data_split[key]
            for dialogue in data:
                for turn in dialogue['turns']:
                    dialogue_acts = turn['dialogue_acts']
                    act_list = flatten_acts(dialogue_acts)
                    delex_acts = delexicalize_da(act_list, self.requestable)

                    if turn['speaker'] == 'system':
                        for act in delex_acts:
                            act = "-".join(act)
                            if act not in system_dict:
                                system_dict[act] = 1
                            else:
                                system_dict[act] += 1
                    else:
                        for act in delex_acts:
                            act = "-".join(act)
                            if act not in user_dict:
                                user_dict[act] = 1
                            else:
                                user_dict[act] += 1

        for key in deepcopy(system_dict):
            if system_dict[key] < frequency_threshold:
                del system_dict[key]

        for key in deepcopy(user_dict):
            if user_dict[key] < frequency_threshold:
                del user_dict[key]

        with open("sys_da_voc.txt", "w") as f:
            system_acts = list(system_dict.keys())
            system_acts.sort()
            for act in system_acts:
                f.write(act + "\n")
        with open("user_da_voc.txt", "w") as f:
            user_acts = list(user_dict.keys())
            user_acts.sort()
            for act in user_acts:
                f.write(act + "\n")
        print("Saved new action dict.")

        self.da_voc = system_acts
        self.da_voc_opp = user_acts

    def load_actions_from_ontology(self):

        self.da_voc = []
        self.da_voc_opp = []
        for act_type in self.ontology['dialogue_acts']:
            for act in self.ontology['dialogue_acts'][act_type]:
                system = act['system']
                user = act['user']
                if system:
                    system_acts_with_value = self.add_values_to_act(act['domain'], act['intent'], act['slot'], True)
                    self.da_voc.extend(system_acts_with_value)

                if user:
                    user_acts_with_value = self.add_values_to_act(act['domain'], act['intent'], act['slot'], False)
                    self.da_voc_opp.extend(user_acts_with_value)

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)
        self.opp2vec = dict((a, i) for i, a in enumerate(self.da_voc_opp))
        self.da_opp_dim = len(self.da_voc_opp)

        print(f"Dimension of system actions: {self.da_dim}")
        print(f"Dimension of user actions: {self.da_opp_dim}")

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

    def add_values_to_act(self, domain, intent, slot, system):
        '''
        The ontology does not contain information about the value of an act. This method will add the value and
        is based on how it is created in MultiWOZ. This might need to be changed for other datasets such as SGD.
        '''

        if intent == 'request':
            return [f"{domain}-{intent}-{slot}-?"]

        if slot == '':
            return [f"{domain}-{intent}-none-none"]

        if system:
            if intent in ['recommend', 'select', 'inform']:
                return [f"{domain}-{intent}-{slot}-{i}" for i in range(1, 4)]
            else:
                return [f"{domain}-{intent}-{slot}-1"]
        else:
            return [f"{domain}-{intent}-{slot}-1"]

    def init_domain_active_dict(self):
        domain_active_dict = {}
        for domain in self.domains:
            if domain == 'general':
                continue
            domain_active_dict[domain] = False
        return domain_active_dict

    def set_seed(self, seed):
        np.random.seed(seed)

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
        constraints = [[slot, value] for slot, value in self.state[domain].items() if value] \
            if domain in self.state else []
        return self.db.query(domain.lower(), constraints, topk=10)

    def find_nooffer_slot(self, domain):
        """
        Function used to find which user constraint results in no entities being found

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


if __name__ == '__main__':
    vector = VectorBase()

