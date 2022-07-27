import torch
import torch.nn as nn
import logging

from torch.nn.utils.rnn import pad_sequence
from .noisy_linear import NoisyLinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCritic(nn.Module):

    def __init__(self, node_embedder, encoder, cls_dim=128, independent=True, enc_nhead=2, noisy_linear=False,
                 **kwargs):
        super(EncoderCritic, self).__init__()

        self.node_embedder = node_embedder
        self.encoder = encoder
        #self.cls = torch.nn.Parameter(torch.randn(cls_dim), requires_grad=True).to(DEVICE)
        self.cls = torch.randn(cls_dim)
        self.cls.requires_grad = True
        self.cls = torch.nn.Parameter(self.cls)
        if noisy_linear:
            logging.info("EncoderCritic: We use noisy linear layers.")
            self.linear = NoisyLinear(cls_dim, 1).to(DEVICE)
        else:
            self.linear = torch.nn.Linear(cls_dim, 1).to(DEVICE)
        self.num_heads = enc_nhead

        logging.info(f"Initialised critic. Independent: {independent}")

    def forward(self, descriptions_list, value_list):
        # return output of cls token
        return self.linear(self.encode_kg(descriptions_list, value_list)[:, 0, :])

    def encode_kg(self, descriptions_list, value_list):
        #encoder_mask = self.compute_mask_extended(kg_list)
        encoder_mask = self.compute_mask(descriptions_list)
        embedded_nodes = self.embedd_nodes(descriptions_list, value_list)
        padded_graphs = pad_sequence(embedded_nodes, batch_first=False).to(DEVICE)
        encoded_nodes, att_weights = self.encoder(padded_graphs, src_key_padding_mask=encoder_mask)
        # size [num_graphs, max_num_nodes, enc_input_dim]
        return encoded_nodes.permute(1, 0, 2)

    def embedd_nodes(self, descriptions_list, value_list):
        kg_sizes = [len(kg) for kg in descriptions_list]

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
            embedded_nodes.append(
                torch.cat([self.cls.unsqueeze(0), flat_embedded_nodes[counter:counter + size, :]], dim=0))
            counter += size
        return embedded_nodes

    def compute_mask(self, kg_list, all=True):
        # we add 1 for the cls_node in every graph
        kg_sizes = [len(kg) + 1 for kg in kg_list]
        max_size = max(kg_sizes)

        attention_mask = torch.ones((len(kg_list), max_size))

        for idx, size in enumerate(kg_sizes):
            if not all:
                attention_mask[idx, idx] = 0
            else:
                attention_mask[idx, :size] = torch.zeros(size)

        return attention_mask.bool().to(DEVICE)

    def compute_mask_extended(self, kg_list):

        kg_sizes = [len(kg) + 1 for kg in kg_list]
        max_size = max(kg_sizes)
        attention_mask = torch.ones((len(kg_list), max_size, max_size))

        domain_list = []
        for kg in kg_list:
            node_dict = {}
            for idx, node in enumerate(kg):
                domain = node['domain']
                if domain not in node_dict:
                    node_dict[domain] = torch.ones(max_size)
                    node_dict[domain][idx + 1] = 0
                else:
                    node_dict[domain][idx + 1] = 0

            domain_list.append(node_dict)

        for idx, kg in enumerate(kg_list):
            for idx_n, node in enumerate(kg):
                domain = node['domain']
                attention_mask[idx, idx_n + 1] = domain_list[idx][domain]

            attention_mask[idx, 0, :len(kg) + 1] = torch.zeros(len(kg) + 1)
            pad_size = max_size - (len(kg) + 1)
            attention_mask[idx, len(kg) + 1:, :] = torch.zeros(pad_size, max_size)

        attention_mask = attention_mask.repeat(self.num_heads, 1, 1)

        return attention_mask.bool().to(DEVICE)
