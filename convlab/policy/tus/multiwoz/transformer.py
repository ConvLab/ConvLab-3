import json
import math
import os
from copy import deepcopy
from random import choice

import numpy as np
import torch

from convlab.policy.policy import Policy
from convlab.policy.tus.multiwoz.util import int2onehot
from torch import nn
from torch.autograd import Variable
from torch.nn import (GRU, CrossEntropyLoss, LayerNorm, Linear,
                      TransformerEncoder, TransformerEncoderLayer)


# TODO masking
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class EncodeLayer(torch.nn.Module):
    def __init__(self, config):
        super(EncodeLayer, self).__init__()
        self.config = config
        self.num_token = self.config["num_token"]
        self.embed_dim = self.config["hidden"]
        transform_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.config["nhead"],
            dim_feedforward=self.config["hidden"],
            activation='gelu')

        self.norm_1 = LayerNorm(self.embed_dim)
        self.encoder = TransformerEncoder(
            encoder_layer=transform_layer,
            num_layers=self.config["num_transform_layer"],
            norm=self.norm_1)
        self.norm_2 = LayerNorm(self.embed_dim)
        self.use_gelu = self.config.get("gelu", False)
        if self.use_gelu:
            self.fc_1 = Linear(self.embed_dim, self.embed_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(self.config["dropout"])
            self.fc_2 = Linear(self.embed_dim, self.embed_dim)
        else:
            self.fc = Linear(self.embed_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.use_gelu:
            self.fc_1.weight.data.uniform_(-initrange, initrange)
            self.fc_2.weight.data.uniform_(-initrange, initrange)
        else:
            self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask):
        x = self.encoder(x, src_key_padding_mask=mask)
        return x


class TransformerActionPrediction(torch.nn.Module):
    def __init__(self, config):
        super(TransformerActionPrediction, self).__init__()
        self.config = config
        self.num_transformer = self.config["num_transformer"]
        self.embed_dim = self.config["embed_dim"]
        self.out_dim = self.config["out_dim"]
        self.hidden = self.config["hidden"]
        self.softmax = nn.Softmax(dim=-1)
        self.num_token = self.config["num_token"]
        self.embed_linear = Linear(self.embed_dim, self.hidden)
        self.position = PositionalEncoding(self.hidden, self.config)
        self.encoder_layers = get_clones(
            EncodeLayer(self.config), N=self.num_transformer)

        self.norm_1 = LayerNorm(self.hidden)
        self.decoder = Linear(self.hidden, self.out_dim)
        self.norm_2 = LayerNorm(self.out_dim)

        weight = [1.0] * self.out_dim
        for i in range(self.out_dim):
            weight[i] /= self.config["weight_factor"][i]

        weight = torch.tensor(weight)
        self.loss = CrossEntropyLoss(weight=weight, ignore_index=-1)
        self.rl_loss = CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.pick_loss = CrossEntropyLoss()
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.domain_loss = nn.BCEWithLogitsLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed_linear.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_feat, mask, label=None, domain_label=None):
        mask = mask.bool()
        src = self.embed_linear(input_feat) * math.sqrt(self.hidden)
        src = self.position(src)
        src = src.permute(1, 0, 2)
        for i in range(self.num_transformer):
            src = self.encoder_layers[i](src, mask)

        out_src = self.norm_1(src)
        out_src = src + out_src

        out_src = out_src.permute(1, 0, 2)

        out = self.decoder(out_src)
        out = self.norm_2(out)

        if label is not None:
            if domain_label is None:
                loss = self.get_loss(out, label)
            else:
                loss = self.get_loss(out, label) + \
                    self.first_token_loss(out, domain_label)
            return loss, out
        return out

    def get_loss(self, prediction, target):
        # prediction = [batch_size, num_token, out_dim]
        # target = [batch_size, num_token]
        # first token is CLS
        pre = prediction[:, 1: self.num_token + 1, :]
        pre = torch.reshape(pre, (pre.shape[0]*pre.shape[1], pre.shape[-1]))
        l = self.loss(pre, target.view(-1))

        return l

    def first_token_loss(self, prediction, target):
        # prediction = [batch_size, num_token, out_dim]
        # target = [batch_size, num_token]
        pre = prediction[:, 0, :]
        l = self.domain_loss(pre, target)
        return l

    def get_log_prob(self, s, a, action_mask=0):
        # s: [b, s_dim]
        # a: [b, a_dim] = [0, 1, -1, -1, ...]
        # forward to get action probs
        action_mask = action_mask.bool()
        a_weights = self.forward(input_feat=s, mask=action_mask)
        pre = a_weights[:, 1: self.num_token + 1, :]
        pre = pre.permute(0, 2, 1)
        loss = self.rl_loss(pre, a)
        loss = torch.mean(loss, 1) * -1
        # old version
        a_probs = self.softmax(a_weights[:, 1: self.num_token + 1, :])
        select_actions = self.one_hot(a)

        log_probs = torch.log(a_probs) * select_actions
        final_score = log_probs.sum(-1).sum(-1)  # [b, 1]

        # print("get_log_prob", loss, final_score)

        return loss
        # return log_prob.sum(-1, keepdim=True)

    @staticmethod
    def one_hot(labels):
        select_actions = []
        for label in labels:
            batch_select = [int2onehot(l) for l in label]
            select_actions.append(batch_select)
        return torch.tensor(select_actions).bool().to(device=DEVICE)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, config, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(p=dropout)
        self.turn_pos = self.config.get("turn-pos", True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position = torch.div(
            position, self.config["num_token"], rounding_mode="floor")
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        pe1 = torch.zeros(max_len, d_model)
        position1 = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe1[:, 0::2] = torch.sin(position1 * div_term1)
        pe1[:, 1::2] = torch.cos(position1 * div_term1)
        pe1 = pe1.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe1', pe1)

    def forward(self, x):
        if self.turn_pos:
            x = x + self.pe1[:x.size(0), :]*0.5 + self.pe[:x.size(0), :]*0.5
        else:
            x = x + self.pe1[:x.size(0), :]
        # x = torch.cat((x, self.pe, self.pe1), -1)
        return self.dropout(x)
