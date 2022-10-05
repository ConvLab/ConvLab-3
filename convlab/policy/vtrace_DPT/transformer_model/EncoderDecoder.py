from torch.nn.utils.rnn import pad_sequence
from .node_embedder import NodeEmbedderRoberta
from .transformer import TransformerModelEncoder, TransformerModelDecoder
from .action_embedder import ActionEmbedder
from torch.distributions.categorical import Categorical
from .noisy_linear import NoisyLinear
from tqdm import tqdm

import torch
import torch.nn as nn
import sys
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderDecoder(nn.Module):
    '''
    Documentation
    '''
    def __init__(self, enc_input_dim, enc_nhead, enc_d_hid, enc_nlayers, enc_dropout,
                 dec_input_dim, dec_nhead, dec_d_hid, dec_nlayers, dec_dropout,
                 action_embedding_dim, action_dict, domain_embedding_dim, value_embedding_dim,
                 node_embedding_dim, roberta_path="", node_attention=True, max_length=25, semantic_descriptions=True,
                 freeze_roberta=True, use_pooled=False, verbose=False, mean=False, ignore_features=None,
                 only_active_values=False, roberta_actions=False, independent_descriptions=False, need_weights=True,
                 random_matrix=False, distance_metric=False, noisy_linear=False, dataset_name='multiwoz21', **kwargs):
        super(EncoderDecoder, self).__init__()
        self.node_embedder = NodeEmbedderRoberta(node_embedding_dim, freeze_roberta=freeze_roberta,
                                                 use_pooled=use_pooled, roberta_path=roberta_path,
                                                 semantic_descriptions=semantic_descriptions, mean=mean,
                                                 dataset_name=dataset_name).to(DEVICE)
        #TODO: Encoder input dim should be same as projection dim or use another linear layer?
        self.encoder = TransformerModelEncoder(enc_input_dim, enc_nhead, enc_d_hid, enc_nlayers, enc_dropout, need_weights).to(DEVICE)
        self.decoder = TransformerModelDecoder(action_embedding_dim, dec_nhead, dec_d_hid, dec_nlayers, dec_dropout, need_weights).to(DEVICE)
        if not roberta_actions:
            self.action_embedder = ActionEmbedder(action_dict, domain_embedding_dim, value_embedding_dim,
                                                  action_embedding_dim,
                                                  random_matrix=random_matrix,
                                                  distance_metric=distance_metric).to(DEVICE)
        else:
            self.action_embedder = ActionEmbedder(action_dict, domain_embedding_dim, value_embedding_dim,
                                                  action_embedding_dim, node_embedder=self.node_embedder,
                                                  random_matrix=random_matrix,
                                                  distance_metric=distance_metric).to(DEVICE)
        #TODO: Ignore features for better robustness and simulating absence of certain information
        self.ignore_features = ignore_features
        self.node_attention = node_attention
        self.freeze_roberta = freeze_roberta
        self.max_length = max_length
        self.verbose = verbose
        self.only_active_values = only_active_values
        self.num_heads = enc_nhead
        self.action_embedding_dim = action_embedding_dim
        # embeddings for "domain", intent", "slot" and "start"
        self.embedding = nn.Embedding(4, action_embedding_dim).to(DEVICE)
        self.info_dict = {}

        if noisy_linear:
            logging.info("EncoderDecoder: We use noisy linear layers.")
            self.action_projector = NoisyLinear(dec_input_dim, action_embedding_dim).to(DEVICE)
            self.current_domain_predictor = NoisyLinear(dec_input_dim, 1).to(DEVICE)
        else:
            self.action_projector = torch.nn.Linear(dec_input_dim, action_embedding_dim).to(DEVICE)
            self.current_domain_predictor = torch.nn.Linear(dec_input_dim, 1).to(DEVICE)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

        self.num_book = 0
        self.num_nobook = 0
        self.num_selected = 0

    def get_current_domain_mask(self, kg_list, current=True):

        current_domains = self.get_current_domains(kg_list)
        current_domain_mask = self.action_embedder.get_current_domain_mask(current_domains[0], current=current).to(DEVICE)
        return current_domain_mask

    def get_descriptions_and_values(self, kg_list):

        description_idx_list = self.node_embedder.description_2_idx(kg_list[0]).to(DEVICE)
        value_list = torch.Tensor([node['value'] for node in kg_list[0]]).unsqueeze(1).to(DEVICE)
        return description_idx_list, value_list

    def select_action(self, kg_list, mask=None, eval=False):
        '''
        :param kg_list: A single knowledge graph consisting of a list of nodes
        :return: multi-action
        Will also return tensors that are used for calculating log-probs, i.e. for doing RL training
        '''

        kg_list = [[node for node in kg if node['node_type'] not in self.ignore_features] for kg in kg_list]
        # this is a bug during supervised training that they use ticket instead of people in book information
        kg_list = [[node for node in kg if node['description'] != "user goal-train-ticket"] for kg in kg_list]

        current_domains = self.get_current_domains(kg_list)
        legal_mask = self.action_embedder.get_legal_mask(mask)

        if self.only_active_values:
            kg_list = [[node for node in kg if node['value'] != 0.0] for kg in kg_list]

        description_idx_list, value_list = self.get_descriptions_and_values(kg_list)
        encoded_nodes, att_weights_encoder = self.encode_kg([description_idx_list], [value_list])
        encoded_nodes = encoded_nodes.to(DEVICE)

        active_domains = set([node['domain'].lower() for node in kg_list[0]] + ['general', 'booking'])

        decoder_input = self.embedding(torch.Tensor([3]).long().to(DEVICE)) + self.embedding(torch.Tensor([0]).to(DEVICE).long())
        decoder_input = decoder_input.view(1, 1, -1).to(DEVICE)
        start = True
        action_mask = self.action_embedder.get_action_mask(start=start)
        action_mask = action_mask + legal_mask
        action_mask = action_mask.bool().float()
        action_mask_list = [action_mask]
        action_list = []
        action_list_num = []
        distribution_list = []
        attention_weights_list = []

        current_domain_mask = self.action_embedder.get_current_domain_mask(current_domains[0], current=True).to(DEVICE)
        non_current_domain_mask = self.action_embedder.get_current_domain_mask(current_domains[0], current=False).to(DEVICE)

        domains = [d for d, i in sorted(self.action_embedder.domain_dict.items(), key=lambda item: item[1])]
        domain_indices = [self.action_embedder.small_action_dict[d] for d in domains]

        intents = [d for d, i in sorted(self.action_embedder.intent_dict.items(), key=lambda item: item[1])]
        intent_indices = [self.action_embedder.small_action_dict[d] for d in intents]

        slot_values = [d for d, i in sorted(self.action_embedder.slot_value_dict.items(), key=lambda item: item[1])]
        s_v_indices = [self.action_embedder.small_action_dict[d] for d in slot_values]

        for t in range(self.max_length):
            decoder_output, att_weights_decoder = self.decoder(decoder_input, encoded_nodes.permute(1, 0, 2))
            attention_weights_list.append(att_weights_decoder)
            action_logits = self.action_embedder(self.action_projector(decoder_output))

            if t % 3 == 0:
                # We need to choose a domain
                current_domain_empty = float((len(current_domains[0]) == 0))
                # we mask taking a current domain if there is none
                pick_current_domain_prob = self.sigmoid(
                    self.current_domain_predictor(decoder_output) - current_domain_empty * sys.maxsize)

                # only pick from current domains
                action_logits_current_domain = action_logits - (
                            action_mask + current_domain_mask).bool().float() * sys.maxsize
                action_distribution_current_domain = self.softmax(action_logits_current_domain)
                action_distribution_current_domain = (action_distribution_current_domain * pick_current_domain_prob)

                # only pick from non-current domains
                action_logits_non_current_domain = action_logits - (
                            action_mask + non_current_domain_mask).bool().float() * sys.maxsize
                action_distribution_non_current_domain = self.softmax(action_logits_non_current_domain)
                action_distribution_non_current_domain = (
                            action_distribution_non_current_domain * (1.0 - pick_current_domain_prob))

                action_distribution = action_distribution_non_current_domain + action_distribution_current_domain
                action_distribution = (action_distribution / action_distribution.sum(dim=-1, keepdim=True)).squeeze(-1)

            else:
                action_logits = action_logits - action_mask * sys.maxsize
                action_distribution = self.softmax(action_logits).squeeze(-1)

            if not eval or t % 3 != 0:
                dist = Categorical(action_distribution)
                rand_state = torch.random.get_rng_state()
                action = dist.sample().tolist()[-1]
                torch.random.set_rng_state(rand_state)
                semantic_action = self.action_embedder.small_action_dict_reversed[action[-1]]
                action_list.append(semantic_action)
                action_list_num.append(action[-1])
            else:
                action = action_distribution[-1, -1, :]
                action = torch.argmax(action).item()
                semantic_action = self.action_embedder.small_action_dict_reversed[action]
                action_list.append(semantic_action)
                action_list_num.append(action)

            #prepare for next step
            next_input = self.action_embedder.action_projector(self.action_embedder.action_embeddings[action]).view(1, 1, -1) + \
                         self.embedding(torch.Tensor([(t + 1) % 3]).to(DEVICE).long())
            decoder_input = torch.cat([decoder_input, next_input], dim=0)

            if t % 3 == 0:
                # We chose a domain
                action_mask_restricted = action_mask[domain_indices]
                domain_dist = action_distribution[0, -1, :][domain_indices]
                distribution_list.append(
                    [semantic_action, dict((domain, (distri, m)) for domain, distri, m in
                                           zip(domains, domain_dist, action_mask_restricted))])

                if semantic_action == 'eos':
                    break
                chosen_domain = semantic_action
                # focus only on the chosen domain information

                action_mask = self.action_embedder.get_action_mask(domain=semantic_action, start=False)
                action_mask = action_mask + self.action_embedder.get_legal_mask(mask, domain=semantic_action)
                action_mask = action_mask.bool().float()
            elif t % 3 == 1:
                # We chose an intent
                if semantic_action == "book":
                    self.num_book += 1
                if semantic_action == "nobook":
                    self.num_nobook += 1

                action_mask = self.action_embedder.get_action_mask(domain=chosen_domain,
                                                                   intent=semantic_action, start=False)
                action_mask = action_mask + self.action_embedder.get_legal_mask(mask, domain=chosen_domain,
                                                                                intent=semantic_action)
                action_mask = action_mask.bool().float()
                #intent_dist = action_distribution[0, -1, :][intent_indices]
                #distribution_list.append(
                #    [semantic_action, dict((intent, (distri, m)) for intent, distri, m in
                 #                          zip(intents, intent_dist, action_mask_restricted))])
            else:
                # We chose a slot-value pair
                action_mask = self.action_embedder.get_action_mask(start=False)
                action_mask = action_mask + self.action_embedder.get_legal_mask(mask)
                action_mask = action_mask.bool().float()

            action_mask_list.append(action_mask)

        self.num_selected += 1

        if action_list[-1] != 'eos':
            action_mask_list = action_mask_list[:-1]

        self.info_dict["kg"] = kg_list[0]
        self.info_dict["small_act"] = torch.Tensor(action_list_num)
        self.info_dict["action_mask"] = torch.stack(action_mask_list)
        self.info_dict["description_idx_list"] = description_idx_list
        self.info_dict["value_list"] = value_list
        self.info_dict["semantic_action"] = action_list
        self.info_dict["current_domain_mask"] = current_domain_mask
        self.info_dict["non_current_domain_mask"] = non_current_domain_mask
        self.info_dict["active_domains"] = active_domains
        self.info_dict["attention_weights"] = attention_weights_list

        if self.verbose:
            print("NEW SELECTION **************************")
            print(f"KG: {kg_list}")
            print(f"Active Domains: {active_domains}")
            print(f"Semantic Act: {action_list}")
            #print("DISTRIBUTION LIST", distribution_list)
            print("Attention:", attention_weights_list[1][1][1])

        return self.action_embedder.small_action_list_to_real_actions(action_list)

    def get_log_prob(self, actions, action_mask_list, max_length, action_targets,
                 current_domain_mask, non_current_domain_mask, descriptions_list, value_list, no_slots=False):

        action_probs, entropy_probs = self.get_prob(actions, action_mask_list, max_length, action_targets,
                 current_domain_mask, non_current_domain_mask, descriptions_list, value_list)
        log_probs = torch.log(action_probs)

        entropy_probs = torch.where(entropy_probs < 0.00001, torch.ones(entropy_probs.size()).to(DEVICE), entropy_probs)
        entropy_probs = torch.where(entropy_probs > 1.0, torch.ones(entropy_probs.size()).to(DEVICE), entropy_probs)
        entropy = -(entropy_probs * torch.log(entropy_probs)).sum(-1).sum(-1).mean()

        # sometimes a domain will be masked because it is inactive due to labelling error. Will ignore these cases.
        log_probs[log_probs == -float("Inf")] = 0

        if no_slots:
            time_steps = torch.arange(0, max_length)
            slot_steps = torch.where(time_steps % 3 == 2, torch.zeros(max_length), torch.ones(max_length))\
                .view(1, -1).to(DEVICE)
            log_probs *= slot_steps

        return log_probs.sum(-1), entropy

    def get_prob(self, actions, action_mask_list, max_length, action_targets,
                 current_domain_mask, non_current_domain_mask, descriptions_list, value_list):
        if not self.freeze_roberta:
            self.node_embedder.form_embedded_descriptions()

        current_domain_mask = current_domain_mask.unsqueeze(1).to(DEVICE)
        non_current_domain_mask = non_current_domain_mask.unsqueeze(1).to(DEVICE)

        encoded_nodes, att_weights_encoder = self.encode_kg(descriptions_list, value_list)
        encoder_mask = self.compute_mask(descriptions_list)
        padded_decoder_input, padded_action_targets = self.get_decoder_tensors(actions, max_length, action_targets)
        # produde decoder mask to not attend to future time-steps
        decoder_mask = torch.triu(torch.ones(max_length, max_length) * float('-inf'), diagonal=1)

        decoder_output, att_weights_decoder = self.decoder(padded_decoder_input.permute(1, 0, 2).to(DEVICE),
                                                           encoded_nodes.permute(1, 0, 2).to(DEVICE),
                                                           decoder_mask.to(DEVICE), encoder_mask.to(DEVICE))

        pick_current_domain_prob = self.sigmoid(self.current_domain_predictor(decoder_output.permute(1, 0, 2)).clone())

        action_logits = self.action_embedder(self.action_projector(decoder_output.permute(1, 0, 2)))

        # do the general mask for intent and slots, domain must be treated separately
        action_logits_general = action_logits - action_mask_list * sys.maxsize
        action_distribution_general = self.softmax(action_logits_general)

        # only pick from current domains
        action_logits_current_domain = action_logits - (action_mask_list + current_domain_mask).bool().float() * sys.maxsize
        action_distribution_current_domain = self.softmax(action_logits_current_domain)
        action_distribution_current_domain = (action_distribution_current_domain * pick_current_domain_prob)

        # only pick from non-current domains
        action_logits_non_current_domain = action_logits - (action_mask_list + non_current_domain_mask).bool().float() * sys.maxsize
        action_distribution_non_current_domain = self.softmax(action_logits_non_current_domain)
        action_distribution_non_current_domain = (action_distribution_non_current_domain * (1.0 - pick_current_domain_prob))

        action_distribution_domain = action_distribution_non_current_domain + action_distribution_current_domain

        time_steps = torch.arange(0, max_length)
        non_domain_steps = torch.where(time_steps % 3 == 0, torch.zeros(max_length), torch.ones(max_length))\
            .view(1, -1, 1).to(DEVICE)
        domain_steps = torch.where(time_steps % 3 == 0, torch.ones(max_length), torch.zeros(max_length))\
            .view(1, -1, 1).to(DEVICE)

        action_distribution_domain = (action_distribution_domain * domain_steps)
        action_distribution_general = (action_distribution_general * non_domain_steps)
        action_distribution = action_distribution_domain + action_distribution_general
        # make sure it sums up to 1 in every time-step
        action_distribution = (action_distribution / action_distribution.sum(dim=-1, keepdim=True))

        action_probs = action_distribution.gather(-1, padded_action_targets.long().unsqueeze(-1)).squeeze()
        # padded time-steps can have very low probability, taking log can be unstable. This prevents it.
        action_prob_helper = torch.Tensor(
            [[1] * len(actions) + [0] * (max_length - len(actions)) for actions in action_targets]).to(DEVICE)
        action_prob_helper_rev = torch.Tensor(
            [[0] * len(actions) + [1] * (max_length - len(actions)) for actions in action_targets]).to(DEVICE)
        # set all padded time-steps to 0 probability
        action_probs = action_probs * action_prob_helper
        # set padded time-steps to probability 1, so that log will be 0
        action_probs = action_probs + action_prob_helper_rev

        entropy_probs = action_distribution_general * action_prob_helper.unsqueeze(-1) + action_prob_helper_rev.unsqueeze(-1)
        #entropy_probs = entropy_probs + domain_steps
        return action_probs, entropy_probs

    def get_current_domains(self, kg_list):
        current_domains = []
        for kg in kg_list:
            curr_list = []
            for node in kg:
                if node['node_type'] == 'user act':
                    if node['domain'].lower() not in current_domains:
                        curr_list.append(node['domain'].lower())
            current_domains.append(curr_list)
        return current_domains

    def get_decoder_tensors(self, actions, max_length, action_targets):

        # Map the actions to action embeddings that are fed as input to decoder model
        # pad input and remove "eos" token
        padded_decoder_input = torch.stack(
            [torch.cat([act[:-1].to(DEVICE), torch.zeros(max_length - len(act)).to(DEVICE)], dim=-1) for act in action_targets], dim=0) \
            .to(DEVICE).long()

        padded_action_targets = torch.stack(
            [torch.cat([act.to(DEVICE), torch.zeros(max_length - len(act)).to(DEVICE)], dim=-1) for act in action_targets], dim=0) \
            .to(DEVICE)

        decoder_input = self.action_embedder.action_embeddings[padded_decoder_input]
        decoder_input = self.action_embedder.action_projector(decoder_input)
        # Add "start" token
        start_input = self.embedding(torch.Tensor([3]).to(DEVICE).long()).to(DEVICE).repeat(len(actions), 1, 1)
        decoder_input = torch.cat([start_input, decoder_input], dim=1)
        # Add "domain", "intent" or "slot" token to input so model knows what to predict
        type_tokens = self.embedding(torch.remainder(torch.Tensor(range(max_length)).to(DEVICE), 3).long())
        decoder_input += type_tokens

        return decoder_input, padded_action_targets

    def encode_kg(self, descriptions_list, value_list):
        #encoder_mask = self.compute_mask_extended(kg_list)
        encoder_mask = self.compute_mask(descriptions_list)
        embedded_nodes = self.embedd_nodes(descriptions_list, value_list)
        padded_graphs = pad_sequence(embedded_nodes, batch_first=False).to(DEVICE)
        encoded_nodes, att_weights = self.encoder(padded_graphs, src_key_padding_mask=encoder_mask)
        # size [num_graphs, max_num_nodes, enc_input_dim]

        return encoded_nodes.permute(1, 0, 2), att_weights

    def embedd_nodes(self, descriptions_list, value_list):
        kg_sizes = [len(descr_list) for descr_list in descriptions_list]

        # we view the kg_list as one huge knowledge graph to embed all nodes simultaneously
        flattened_descriptions = torch.stack(
            [descr_idx for descr_list in descriptions_list for descr_idx in descr_list]).to(DEVICE)
        flattened_values = torch.stack(
            [value for values in value_list for value in values])
        flat_embedded_nodes = self.node_embedder(flattened_descriptions, flattened_values).to(DEVICE)

        #now get back the individual knowledge graphs
        embedded_nodes = []
        counter = 0
        for size in kg_sizes:
            embedded_nodes.append(flat_embedded_nodes[counter:counter + size, :])
            counter += size
        return embedded_nodes

    def compute_mask(self, descriptions_list):
        kg_sizes = [len(descr_list) for descr_list in descriptions_list]
        max_size = max(kg_sizes)
        attention_mask = torch.ones((len(descriptions_list), max_size))

        for idx, size in enumerate(kg_sizes):
            attention_mask[idx, :size] = torch.zeros(size)

        return attention_mask.bool().to(DEVICE)

    def compute_mask_extended(self, kg_list):

        kg_sizes = [len(kg) for kg in kg_list]
        max_size = max(kg_sizes)
        attention_mask = torch.ones((len(kg_list), max_size, max_size))

        domain_list = []
        for kg in kg_list:
            node_dict = {}
            for idx, node in enumerate(kg):
                domain = node['domain']
                if domain not in node_dict:
                    node_dict[domain] = torch.ones(max_size)
                    node_dict[domain][idx] = 0
                else:
                    node_dict[domain][idx] = 0

            domain_list.append(node_dict)

        for idx, kg in enumerate(kg_list):
            for idx_n, node in enumerate(kg):
                domain = node['domain']
                attention_mask[idx, idx_n] = domain_list[idx][domain]
            pad_size = max_size - len(kg)
            attention_mask[idx, len(kg):, :] = torch.zeros(pad_size, max_size)

        attention_mask = attention_mask.repeat(self.num_heads, 1, 1)

        return attention_mask.bool().to(DEVICE)

    def get_action_masks(self, actions):
        # active domains
        # active_domain_list = [set([node['domain'].lower() for node in kg] + ['general', 'booking']) for kg in kg_list]
        # print("active domain list", active_domain_list)

        action_targets = [self.action_embedder.real_action_to_small_action_list(act) for act in actions]
        action_lengths = [len(actions) for actions in action_targets]
        max_length = max(action_lengths)

        semantic_acts = [self.action_embedder.real_action_to_small_action_list(act, semantic=True) for act in actions]
        action_mask_list = []
        decoder_encoder_mask_list = []
        for i, act_sequence in tqdm(enumerate(semantic_acts)):
            action_mask = [self.action_embedder.get_action_mask(start=True)]

            for t, act in enumerate(act_sequence):

                if t % 3 == 0:
                    # We chose a domain
                    if act == 'eos':
                        break
                    chosen_domain = act
                    # focus only on the chosen domain information
                    # TODO: Decoder encoder mask is unfinished, but I need to modify the self-attention mask for that
                    # decoder_encoder_mask = [0 if node['domain'] in ['booking', 'general', chosen_domain] else 1
                    #                        for node in kg_list[i]]
                    # decoder_encoder_mask.append(decoder_encoder_mask)

                    action_mask.append(self.action_embedder.get_action_mask(domain=act, start=False))
                elif t % 3 == 1:
                    # We chose an intent
                    action_mask.append(self.action_embedder.get_action_mask(domain=chosen_domain,
                                                                            intent=act, start=(t == 0)))
                else:
                    # We chose a slot-value pair
                    action_mask.append(self.action_embedder.get_action_mask(start=False))

            # pad action mask to get list of max_length
            action_mask = torch.cat([
                torch.stack(action_mask).to(DEVICE),
                torch.zeros(max_length - len(action_mask), len(self.action_embedder.small_action_dict)).to(DEVICE)],
                dim=0)
            action_mask_list.append(action_mask)

        action_mask_list = torch.stack(action_mask_list).to(DEVICE)
        # print("semantic acts:", semantic_acts)
        return action_mask_list, max_length


