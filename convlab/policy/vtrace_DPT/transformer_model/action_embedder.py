import os
import torch
import torch.nn as nn
import logging
import json

from copy import deepcopy
from convlab.policy.vtrace_DPT.transformer_model.noisy_linear import NoisyLinear


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionEmbedder(nn.Module):
    '''
    Obtains the action-dictionary with all actions and creates embeddings for domain, intent and slot-value pairs
    The embeddings are used for creating the domain, intent and slot-value actions in the EncoderDecoder
    '''
    def __init__(self, action_dict, embedding_dim, value_embedding_dim, action_embedding_dim, node_embedder=None,
                 random_matrix=False, distance_metric=False):
        super(ActionEmbedder, self).__init__()

        self.domain_dict, self.intent_dict, self.slot_dict, self.value_dict, self.slot_value_dict \
            = self.create_dicts(action_dict)

        #EOS token is considered a "domain"
        self.action_dict = dict((key.lower(), value) for key, value in action_dict.items())
        self.action_dict_reversed = dict((value, key) for key, value in self.action_dict.items())
        self.embed_domain = torch.randn(len(self.domain_dict), embedding_dim)
        self.embed_intent = torch.randn(len(self.intent_dict), embedding_dim)
        self.embed_slot = torch.randn(len(self.slot_dict), embedding_dim - value_embedding_dim)
        self.embed_value = torch.randn(len(self.value_dict), value_embedding_dim)
        self.embed_rest = torch.randn(1, embedding_dim)     #Pad token
        self.use_random_matrix = random_matrix
        self.distance_metric = distance_metric
        self.forbidden_domains = []

        if not node_embedder:
            logging.info("We train action embeddings from scratch.")
            self.action_embeddings, self.small_action_dict = self.create_action_embeddings(embedding_dim)
            self.action_embeddings.requires_grad = True
            self.action_embeddings = nn.Parameter(self.action_embeddings)
        else:
            logging.info("We use Roberta to embed actions.")
            self.dataset_name = node_embedder.dataset_name
            self.create_action_embeddings_roberta(node_embedder)
            self.action_embeddings.requires_grad = False
            embedding_dim = 768

        #logging.info(f"Small Action Dict: {self.small_action_dict}")

        self.small_action_dict_reversed = dict((value, key) for key, value in self.small_action_dict.items())

        self.linear = torch.nn.Linear(embedding_dim, action_embedding_dim).to(DEVICE)
        #self.linear = NoisyLinear(embedding_dim, action_embedding_dim).to(DEVICE)
        self.random_matrix = torch.randn(embedding_dim, action_embedding_dim).to(DEVICE) / \
                             torch.sqrt(torch.Tensor([768])).to(DEVICE)

    def action_projector(self, actions):
        if self.use_random_matrix:
            return torch.matmul(actions, self.random_matrix).to(DEVICE)
        else:
            return self.linear(actions)

    def forward(self, state):
        # state [batch-size, action_dim], self.action_embeddings [num_actions, embedding_dim]
        action_embeddings = self.action_projector(self.action_embeddings)

        if not self.distance_metric:
            # We use scalar product for similarity
            output = torch.matmul(state, action_embeddings.permute(1, 0))
        else:
            # We use distance metric for similarity as in SUMBT
            output = -torch.cdist(state, action_embeddings, p=2)

        return output

    def get_legal_mask(self, legal_mask, domain="", intent=""):

        if legal_mask is None:
            return torch.zeros(len(self.small_action_dict)).to(DEVICE)

        action_mask = torch.ones(len(self.small_action_dict))
        if not domain:
            for domain in self.domain_dict:
                # check whether we can use that domain, at the moment we want to allow all domains
                action_mask[self.small_action_dict[domain]] = 0
        elif not intent:
            # Domain was selected, check intents that are allowed
            for intent in self.intent_dict:
                domain_intent = f"{domain}-{intent}"
                for idx, not_allow in enumerate(legal_mask):
                    semantic_act = self.action_dict_reversed[idx]
                    if domain_intent in semantic_act and not_allow == 0:
                        action_mask[self.small_action_dict[intent]] = 0
                        break
        else:
            # Selected domain and intent, need slot-value
            for slot_value in self.slot_value_dict:
                domain_intent_slot = f"{domain}-{intent}-{slot_value}"
                for idx, not_allow in enumerate(legal_mask):
                    semantic_act = self.action_dict_reversed[idx]
                    if domain_intent_slot in semantic_act and not_allow == 0:
                        action_mask[self.small_action_dict[slot_value]] = 0
                        break

        return action_mask.to(DEVICE)

    def get_action_mask(self, domain=None, intent="", start=False):

        action_mask = torch.ones(len(self.small_action_dict))

        # This is for predicting end of sequence token <eos>
        if not start and domain is None:
            action_mask[self.small_action_dict['eos']] = 0

        if domain is None:
            #TODO: I allow all domains now for checking supervised training
            for domain in self.domain_dict:
                if domain not in self.forbidden_domains:
                    action_mask[self.small_action_dict[domain]] = 0
            if start:
                action_mask[self.small_action_dict['eos']] = 1
            # Only active domains can be selected
            #for domain in active_domains:
            #    action_mask[self.small_action_dict[domain]] = 0

        elif not intent:
            # Domain was selected, need intent now
            for intent in self.intent_dict:
                domain_intent = f"{domain}-{intent}"
                valid = self.is_valid(domain_intent + "-")
                if valid:
                    action_mask[self.small_action_dict[intent]] = 0
        else:
            # Selected domain and intent, need slot-value
            for slot_value in self.slot_value_dict:
                domain_intent_slot = f"{domain}-{intent}-{slot_value}"
                valid = self.is_valid(domain_intent_slot)
                if valid:
                    action_mask[self.small_action_dict[slot_value]] = 0

        assert not torch.equal(action_mask, torch.ones(len(self.small_action_dict)))

        return action_mask.to(DEVICE)

    def get_current_domain_mask(self, current_domains, current=True):

        action_mask = torch.ones(len(self.small_action_dict))
        if current:
            for domain in current_domains:
                action_mask[self.small_action_dict[domain]] = 0
        else:
            for domain in self.domain_dict:
                if domain not in current_domains:
                    action_mask[self.small_action_dict[domain]] = 0

        return action_mask.to(DEVICE)

    def is_valid(self, part_action):

        for act in self.action_dict:
            if act.startswith(part_action):
                return True

        return False

    def create_action_embeddings(self, embedding_dim):

        action_embeddings = torch.zeros((len(self.domain_dict) + len(self.intent_dict) + len(self.slot_value_dict) + 1,
                                         embedding_dim))

        small_action_dict = {}
        for domain, idx in self.domain_dict.items():
            action_embeddings[len(small_action_dict)] = self.embed_domain[idx]
            small_action_dict[domain] = len(small_action_dict)
        for intent, idx in self.intent_dict.items():
            action_embeddings[len(small_action_dict)] = self.embed_intent[idx]
            small_action_dict[intent] = len(small_action_dict)
        for slot_value in self.slot_value_dict:
            slot, value = slot_value.split("-")
            slot_idx = self.slot_dict[slot]
            value_idx = self.value_dict[value]
            action_embeddings[len(small_action_dict)] = torch.cat(
                (self.embed_slot[slot_idx], self.embed_value[value_idx]))
            small_action_dict[slot_value] = len(small_action_dict)

        action_embeddings[len(small_action_dict)] = self.embed_rest[0]      #add the PAD token
        small_action_dict['pad'] = len(small_action_dict)
        return action_embeddings.to(DEVICE), small_action_dict

    def create_action_embeddings_roberta(self, node_embedder):

        action_embeddings = []

        small_action_dict = {}
        for domain, idx in self.domain_dict.items():
            action_embeddings.append(domain)
            small_action_dict[domain] = len(small_action_dict)
        for intent, idx in self.intent_dict.items():
            action_embeddings.append(intent)
            small_action_dict[intent] = len(small_action_dict)
        for slot_value in self.slot_value_dict:
            slot, value = slot_value.split("-")
            action_embeddings.append(f"{slot} {value}")
            small_action_dict[slot_value] = len(small_action_dict)

        action_embeddings.append("pad")     #add the PAD token
        small_action_dict['pad'] = len(small_action_dict)

        action_embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              f'action_embeddings_{self.dataset_name}.pt')
        small_action_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              f'small_action_dict_{self.dataset_name}.json')

        if os.path.exists(action_embeddings_path):
            self.action_embeddings = torch.load(action_embeddings_path).to(DEVICE)
        else:
            self.action_embeddings = node_embedder.embed_sentences(action_embeddings).to(DEVICE)
            torch.save(self.action_embeddings, action_embeddings_path)

        if os.path.exists(small_action_dict_path):
            self.small_action_dict = json.load(open(small_action_dict_path, 'r'))
        else:
            self.small_action_dict = small_action_dict
            with open(small_action_dict_path, 'w') as f:
                json.dump(self.small_action_dict, f)

        self.small_action_dict = small_action_dict

    def create_dicts(self, action_dict):
        domain_dict = {}
        intent_dict = {}
        slot_dict = {}
        value_dict = {}
        slot_value_dict = {}
        for action in action_dict:
            domain, intent, slot, value = [act.lower() for act in action.split('-')]
            if domain not in domain_dict:
                domain_dict[domain] = len(domain_dict)
            if intent not in intent_dict:
                intent_dict[intent] = len(intent_dict)
            if slot not in slot_dict:
                slot_dict[slot] = len(slot_dict)
            if value not in value_dict:
                value_dict[value] = len(value_dict)
            if slot + "-" + value not in slot_value_dict:
                slot_value_dict[slot + "-" + value] = len(slot_value_dict)

        domain_dict['eos'] = len(domain_dict)

        return domain_dict, intent_dict, slot_dict, value_dict, slot_value_dict

    def small_action_list_to_real_actions(self, small_action_list):

        #print("SMALL ACTION LIST:", small_action_list)
        action_vector = torch.zeros(len(self.action_dict))
        act_string = ""
        for idx, act in enumerate(small_action_list):
            if act == 'eos':
                break

            if idx % 3 != 2:
                act_string += f"{act}-"
            else:
                act_string += act
                action_vector[self.action_dict[act_string]] = 1
                act_string = ""

        return action_vector

    def real_action_to_small_action_list(self, action, semantic=False, permute=False):
        '''
        :param action: [hotel-req-address, taxi-inform-phone]
        :return: [hotel, req, address, taxi, inform, phone, eos]
        '''

        action_list = []
        for idx, i in enumerate(action):
            if i == 1:
                action_list += self.action_dict_reversed[idx].split("-", 2)

        if permute and len(action_list) > 3:
            action_list_new = deepcopy(action_list[-3:]) + deepcopy(action_list[:-3])
            action_list = action_list_new
        action_list.append("eos")

        if semantic:
            return action_list

        action_list = [self.small_action_dict[act] for act in action_list]
        return action_list