# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import logging

from copy import deepcopy
from convlab.policy.vec import Vector
from convlab.util.custom_util import flatten_acts, timeout
from convlab.util.multiwoz.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da
from convlab.util import load_ontology, load_database, load_dataset


root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)


class VectorBase(Vector):

    def __init__(self, dataset_name='multiwoz21', character='sys', use_masking=False, manually_add_entity_names=False,
                 always_inform_booking_reference=True, seed=0):

        super().__init__()

        logging.info(f"Vectorizer: Data set used is {dataset_name}")
        self.set_seed(seed)
        self.ontology = load_ontology(dataset_name)
        try:
            # execute to make sure that the database exists or is downloaded otherwise
            if dataset_name == "multiwoz21":
                load_database(dataset_name)
            # the following two lines are needed for pickling correctly during multi-processing
            exec(f'from data.unified_datasets.{dataset_name}.database import Database')
            self.db = eval('Database()')
            self.db_domains = self.db.domains
        except Exception as e:
            self.db = None
            self.db_domains = []
            print(f"VectorBase: {e}")

        self.dataset_name = dataset_name
        self.max_actionval = {}
        self.use_mask = use_masking
        self.use_add_name = manually_add_entity_names
        self.always_inform_booking_reference = always_inform_booking_reference
        self.reqinfo_filler_action = None
        self.character = character
        self.requestable = ['request']
        self.informable = ['inform', 'recommend']

        self.load_attributes()
        self.get_state_dim()
        print(f"State dimension: {self.state_dim}")

    def load_attributes(self):

        self.domains = list(self.ontology['domains'].keys())
        self.domains.sort()

        self.previous_name_actions = {domain: [] for domain in self.domains}

        self.state = self.ontology['state']
        self.belief_domains = list(self.state.keys())
        self.belief_domains.sort()

        self.load_action_dicts()

    def load_action_dicts(self):

        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'action_dicts/{self.dataset_name}_{type(self).__name__}')
        if not (os.path.exists(os.path.join(dir_path, "sys_da_voc.txt"))
                and os.path.exists(os.path.join(dir_path, "user_da_voc.txt"))):
            print("Load actions from data..")
            self.load_actions_from_data()
        else:
            print("Load actions from file..")
            with open(os.path.join(dir_path, "sys_da_voc.txt")) as f:
                self.da_voc = f.read().splitlines()
            with open(os.path.join(dir_path, "user_da_voc.txt")) as f:
                self.da_voc_opp = f.read().splitlines()

        self.generate_dict()

    def load_actions_from_data(self, frequency_threshold=50):
        """
        Loads the action sets for user and system using a data set.
        The frequency_threshold prohibits adding actions that occur fewer times than this threshold in the data
        (for instance there might be incorrectly labelled actions)
        """

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

        self.da_voc = list(system_dict.keys())
        self.da_voc.sort()
        self.da_voc_opp = list(user_dict.keys())
        self.da_voc_opp.sort()

        self.save_acts_to_txt()

    def save_acts_to_txt(self):
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'action_dicts/{self.dataset_name}_{type(self).__name__}')
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "sys_da_voc.txt"), "w") as f:
            for act in self.da_voc:
                f.write(act + "\n")
        with open(os.path.join(dir_path, "user_da_voc.txt"), "w") as f:
            for act in self.da_voc_opp:
                f.write(act + "\n")

    def load_actions_from_ontology(self):
        """
        Loads the action sets for user and system if an ontology is provided.
        It is recommended to use load_actions_from_data to guarantee consistency with previous results
        """

        self.da_voc = []
        self.da_voc_opp = []
        for act_type in self.ontology['dialogue_acts']:
            for act in self.ontology['dialogue_acts'][act_type]:
                act = eval(act)
                system = act['system']
                user = act['user']
                if system:
                    system_acts_with_value = self.add_values_to_act(
                        act['domain'], act['intent'], act['slot'], True)
                    self.da_voc.extend(system_acts_with_value)

                if user:
                    user_acts_with_value = self.add_values_to_act(
                        act['domain'], act['intent'], act['slot'], False)
                    self.da_voc_opp.extend(user_acts_with_value)

        self.da_voc.sort()
        self.da_voc_opp.sort()

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
        '''
        Can not speak about a domain if that domain is not active.
        A domain is active if the user mentioned it in the current turn or if a slot is filled with a value
        '''
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

            # NoBook/NoOffer-SLOT does not make sense because policy can not know which constraint made offer impossible
            # If one wants to do it, lexicaliser needs to do it
            if intent in ['nobook', 'nooffer'] and slot != 'none':
                mask_list[i] = 1.0

            if "book" in slot and intent == 'inform' and not self.state[domain][slot]:
                mask_list[i] = 1.0

            if domain == 'taxi':
                if slot in self.state['taxi']:
                    if not self.state['taxi'][slot] and intent == 'inform':
                        mask_list[i] = 1.0

        return mask_list

    def compute_entity_mask(self, number_entities_dict):
        '''
        1. If there is no i-th entity in the data base, can not inform/recommend/select on that entity
        2. If there is an entity available, can not say NoOffer or NoBook
        '''
        mask_list = np.zeros(self.da_dim)
        if number_entities_dict is None:
            return mask_list
        for i in range(self.da_dim):
            action = self.vec2act[i]
            domain, intent, slot, value = action.split('-')
            domain_entities = number_entities_dict.get(domain, 1)

            if intent in ['inform', 'select', 'recommend'] and value != None and value != 'none':
                if int(value) > domain_entities:
                    mask_list[i] = 1.0
            if intent in ['nooffer', 'nobook'] and number_entities_dict.get(domain, 0) > 0:
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
        return self.db.query(domain, constraints, topk=10)

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
        constraints = self.state[domain]

        # Leave slots out of constraints to find which slot constraint results in no entities being found
        for constraint_slot in constraints:
            state = [[slot, value] for slot,
                     value in constraints.items() if slot != constraint_slot]
            entities = self.db.query(domain, state, topk=1)
            if entities:
                return constraint_slot

        # If no single slot results in no entities being found try the above with pairs of slots
        slots = [slot for slot in constraints]
        pairs = []
        for i, slot in enumerate(slots):
            for j, slot1 in enumerate(slots):
                if j > i:
                    pairs.append((slot, slot1))

        for constraint_slots in pairs:
            state = [[slot, value] for slot, value in constraints.items() if slot not in constraint_slots]
            entities = self.db.query(domain, state, topk=1)
            if entities:
                return np.random.choice(constraint_slots)

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
            if domain not in entities and domain not in ['general']:
                entities[domain] = self.dbquery_domain(domain)

        # From db query find which slot causes no_offer
        nooffer = [domint for domint in action if 'nooffer' in domint]
        for domint in nooffer:
            domain, intent = domint.split('-')
            slot = self.find_nooffer_slot(domain)
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        # Randomly select booking constraint "causing" no_book
        nobook = [domint for domint in action if 'nobook' in domint]
        for domint in nobook:
            domain, intent = domint.split('-')
            if domain in self.state:
                slots = self.state[domain]
                slots = [slot for slot, i in slots.items()
                         if i and 'book' in slot]
                slots.append('none')
                slot = np.random.choice(slots)
            else:
                slot = 'none'
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        if self.always_inform_booking_reference:
            action = self.add_booking_reference(action)

        # When there is a INFORM(1 name) or OFFER(multiple) action then inform the name
        if self.use_add_name:
            action = self.add_name(action)

        for key in action.keys():
            index = -1
            for [item, idx] in action[key]:
                if index != -1 and index != idx and idx != '?':
                    pass
                    # logging.debug(
                    #    "System is likely refering multiple entities within this turn")
                    # logging.debug(action[key])
                index = idx
        action = lexicalize_da(action, entities, self.state, self.requestable)

        return action

    def add_booking_reference(self, action):
        new_acts = {}
        for domint in action:
            domain, intent = domint.split('-', 1)

            if intent == 'book' and action[domint]:
                ref_domint = f'{domain}-inform'
                if ref_domint not in new_acts:
                    new_acts[ref_domint] = []
                new_acts[ref_domint].append(['ref', '1'])
                if domint not in new_acts:
                    new_acts[domint] = []
                new_acts[domint].append(['none', '1'])
            elif domint in new_acts:
                new_acts[domint] += action[domint]
            else:
                new_acts[domint] = action[domint]

        return new_acts

    def add_name(self, action):

        name_inform = {domain: [] for domain in self.domains}
        # General Inform Condition for Naming
        domains = [domint.split('-', 1)[0] for domint in action]
        domains = list(set([d for d in domains if d not in ['general']]))
        for domain in domains:
            contains_name = False
            if domain == 'none':
                raise NameError('Domain not defined')
            cur_inform = domain + '-inform'
            cur_request = domain + '-request'
            index = -1
            if cur_inform in action:
                # Check if current inform within a domain is accompanied by a name inform
                for [slot, value_id] in action[cur_inform]:
                    if slot == 'name':
                        contains_name = True
                    elif domain == 'train' and slot == 'id':
                        contains_name = True
                    elif domain == 'hospital':
                        contains_name = True
                    elif slot == 'choice' and cur_request in action:
                        contains_name = True

                if not contains_name:
                    # Construct name inform act if name is not contained in acts
                    if domain == 'train':
                        name_inform[domain] = ['id', value_id]
                    else:
                        name_inform[domain] = ['name', value_id]

                    # If name inform act has not been taken before then add to action set
                    if name_inform[domain] != self.previous_name_actions[domain]:
                        action[cur_inform] += [name_inform[domain]]
                        self.previous_name_actions[domain] = name_inform[domain]

        return action

    def pointer(self):
        pointer_vector = np.zeros(6 * len(self.db_domains))
        number_entities_dict = {}
        for domain in self.db_domains:
            entities = self.dbquery_domain(domain)
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
