#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 lubis <lubis@hilbert50>
#
# Distributed under terms of the MIT license.

"""

"""

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pdb
import random
import copy
from collections import namedtuple, deque
from torch.autograd import Variable
from convlab.policy.lava.multiwoz.latent_dialog.enc2dec.encoders import RnnUttEncoder
from convlab.policy.lava.multiwoz.latent_dialog.utils import get_detokenize, cast_type, extract_short_ctx, np2var, LONG, FLOAT
from convlab.policy.lava.multiwoz.latent_dialog.corpora import SYS, EOS, PAD, BOS, DOMAIN_REQ_TOKEN, ACTIVE_BS_IDX, NO_MATCH_DB_IDX, REQ_TOKENS
import dill

class Actor(nn.Module):
    def __init__(self, model, corpus, config):
        super(Actor, self).__init__()
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.config = config

        self.use_gpu = config.use_gpu

        self.embedding = None
        self.is_stochastic = config.is_stochastic
        self.y_size = config.y_size
        if 'k_size' in config:
            self.k_size = config.k_size
            self.is_gauss = False
        else:
            self.max_action = config.max_action if "max_action" in config else None
            self.is_gauss = True

        self.utt_encoder = copy.deepcopy(model.utt_encoder)
        self.c2z = copy.deepcopy(model.c2z)
        if not self.is_gauss:
            self.gumbel_connector = copy.deepcopy(model.gumbel_connector)
        else:
            self.gauss_connector = copy.deepcopy(model.gauss_connector)
            self.gaussian_logprob = model.gaussian_logprob
            self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

        # self.l1 = nn.Linear(self.utt_encoder.output_size, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, config.y_size * config.k_size)

    def forward(self, data_feed, hard=False):
        short_ctx_utts = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        if self.is_gauss:
            q_mu, q_logvar = self.c2z(enc_last)
            # sample_z = q_mu
            if self.is_stochastic:
                sample_z = self.gauss_connector(q_mu, q_logvar)
            else:
                sample_z = q_mu
            logprob_sample_z = self.gaussian_logprob(q_mu, q_logvar, sample_z)
            # joint_logpz = th.sum(logprob_sample_z, dim=1)
            # return self.max_action * torch.tanh(z)
        else:
            logits_qy, log_qy = self.c2z(enc_last)
            qy = F.softmax(logits_qy / 1.0, dim=1)  # (batch_size, vocab_size, )
            log_qy = F.log_softmax(logits_qy, dim=1)  # (batch_size, vocab_size, )

            if self.is_stochastic:
                idx = th.multinomial(qy, 1).detach()
                soft_z = self.gumbel_connector(logits_qy, hard=False)
            else:
                idx = th.argmax(th.exp(log_qy), dim=1, keepdim=True)
                soft_z = th.exp(log_qy)
            sample_z = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_z.scatter_(1, idx, 1.0)
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)

        joint_logpz = th.sum(logprob_sample_z, dim=1)
        # for i in range(logprob_sample_z.shape[0]):
            # print(logprob_sample_z[i])
            # print(joint_logpz[i])
        return joint_logpz, sample_z

class DeterministicGaussianActor(nn.Module):
    def __init__(self, model, corpus, config):
        super(DeterministicGaussianActor, self).__init__()
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.config = config

        self.use_gpu = config.use_gpu

        self.embedding = None
        self.y_size = config.y_size
        self.max_action = config.max_action if "max_action" in config else None
        self.is_gauss = True

        self.utt_encoder = copy.deepcopy(model.utt_encoder)

        self.policy = copy.deepcopy(model.c2z)
        # self.gauss_connector = copy.deepcopy(model.gauss_connector)

    def forward(self, data_feed):
        short_ctx_utts = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        mu, logvar = self.policy(enc_last)
        z = mu
        if self.max_action is not None:
            z =  self.max_action * th.tanh(z)

        return z, mu, logvar

class StochasticGaussianActor(nn.Module):
    def __init__(self, model, corpus, config):
        super(StochasticGaussianActor, self).__init__()
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.config = config

        self.use_gpu = config.use_gpu

        self.embedding = None
        self.y_size = config.y_size
        self.max_action = config.max_action if "max_action" in config else None
        self.is_gauss = True

        self.utt_encoder = copy.deepcopy(model.utt_encoder)
        self.policy = copy.deepcopy(model.c2z)
        self.gauss_connector = copy.deepcopy(model.gauss_connector)

    def forward(self, data_feed, n_z=1):
        short_ctx_utts = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        q_mu, q_logvar = self.policy(enc_last)
        if n_z > 1:
            z = [self.gauss_connector(q_mu, q_logvar) for _ in range(n_z)]
        else:
            z = self.gauss_connector(q_mu, q_logvar)

        return z, q_mu, q_logvar

class RecurrentCritic(nn.Module):
    def __init__(self,cvae, corpus, config, args):
        super(RecurrentLatentCritic, self).__init__()

        # self.vocab = corpus.vocab
        # self.vocab_dict = corpus.vocab_dict
        # self.vocab_size = len(self.vocab)
        # self.bos_id = self.vocab_dict[BOS]
        # self.eos_id = self.vocab_dict[EOS]
        # self.pad_id = self.vocab_dict[PAD]

        self.embedding = None
        self.word_plas = args.word_plas
        self.state_dim = cvae.utt_encoder.output_size
        if self.word_plas:
            self.action_dim = cvae.aux_encoder.output_size
        else:
            self.action_dim = config.y_size #TODO adjust for categorical
        # if "k_size"  in config:
            # if args.embed_z_for_critic:
                # self.action_dim = config.dec_cell_size # for categorical, the action can be embedded
            # else:
        #         self.action_dim *= config.k_size

        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.input_dim = self.state_dim + self.bs_size + self.db_size + self.action_dim
        # self.input_dim = self.state_dim + 50 + self.action_dim
        self.goal_to_critic = args.goal_to_critic
        if self.goal_to_critic:
            raise NotImplementedError

        self.use_gpu = config.use_gpu

        self.state_encoder = copy.deepcopy(cvae.utt_encoder)
        if self.word_plas:
            self.action_encoder = copy.deepcopy(cvae.aux_encoder)
        else:
            self.action_encoder = None

        # self.q11 = nn.Linear(self.state_dim + self.action_dim + 50, 500)
        self.q11 = nn.Linear(self.input_dim, 500)
        self.q12 = nn.Linear(500, 300)
        self.q13 = nn.Linear(300, 100)
        self.q14 = nn.Linear(100, 20)
        self.q15 = nn.Linear(20, 1)

        self.q21 = nn.Linear(self.input_dim, 500)
        self.q22 = nn.Linear(500, 300)
        self.q23 = nn.Linear(300, 100)
        self.q24 = nn.Linear(100, 20)
        self.q25 = nn.Linear(20, 1)

    def forward(self, data_feed, act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, _, _ = self.state_encoder(ctx.unsqueeze(1))
        if self.word_plas:
            resp_summary, _, _ = self.action_encoder(act.unsqueeze(1))
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, resp_summary.squeeze(1)], dim=1)
        else:
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, act], dim=1)

        q1 = self.q11(sa)
        #-
        # q1 = F.relu(self.q12(th.cat([q1, metadata_summary], dim=1)))
        q1 = F.relu(self.q12(q1))
        # q1 = self.q12(q1)
        #-
        # q1 = th.sigmoid(self.q13(q1))
        q1 = F.relu(self.q13(q1))
        # q1 = F.softmax(self.q13(q1))
        #-
        # q1 = th.sigmoid(self.q14(q1))
        q1 = F.relu(self.q14(q1))
        # q1 = self.q14(q1)
        #-
        q1 = self.q15(q1)


        q2 = self.q21(sa)
        #-
        # q2 = F.relu(self.lq22(th.cat([q2, metadata_summary], dim=1)))
        q2 = F.relu(self.q22(q2))
        # q2 = self.q22(q2)
        #-
        # q2 = th.sigmoid(self.q23(q2))
        q2 = F.relu(self.q23(q2))
        # q2 = F.softmax(self.q23(q2))
        #-
        # q2 = th.sigmoid(self.q24(q2))
        q2 = F.relu(self.q24(q2))
        # q2 = self.q24(q2)
        #-
        q2 = self.q25(q2)

        return q1, q2

    def q1(self, data_feed, act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, _, _ = self.state_encoder(ctx.unsqueeze(1))
        if self.word_plas:
            resp_summary, _, _ = self.action_encoder(act.unsqueeze(1))
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, resp_summary.squeeze(0)], dim=1)
        else:
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, act], dim=1)
        
        q1 = self.q11(sa)
        #-
        # q1 = F.relu(self.q12(th.cat([q1, metadata_summary], dim=1)))
        q1 = F.relu(self.q12(q1))
        # q1 = self.q12(q1)
        #-
        # q1 = th.sigmoid(self.q13(q1))
        q1 = F.relu(self.q13(q1))
        # q1 = F.softmax(self.q13(q1))
        #-
        # q1 = th.sigmoid(self.q14(q1))
        q1 = F.relu(self.q14(q1))
        # q1 = self.q14(q1)
        #-
        q1 = self.q15(q1)

        return q1

class SingleRecurrentCritic(nn.Module):
    def __init__(self, cvae, corpus, config, args):
        super(SingleRecurrentCritic, self).__init__()

        # self.vocab = corpus.vocab
        # self.vocab_dict = corpus.vocab_dict
        # self.vocab_size = len(self.vocab)
        # self.bos_id = self.vocab_dict[BOS]
        # self.eos_id = self.vocab_dict[EOS]
        # self.pad_id = self.vocab_dict[PAD]

        if "gauss" in args.sv_config_path:
            self.is_gauss = True
        else:
            self.is_gauss = False
        self.embedding = None
        self.word_plas = args.word_plas
        self.state_dim = cvae.utt_encoder.output_size
        if self.word_plas:
            self.action_dim = cvae.aux_encoder.output_size
        else:
            if self.is_gauss:
                self.action_dim = config.y_size 
            else:
                if args.embed_z_for_critic:
                    self.action_dim = config.dec_cell_size
                else:
                    self.action_dim = config.y_size * config.k_size

        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.input_dim = self.state_dim + self.bs_size + self.db_size + self.action_dim

        self.goal_to_critic = args.goal_to_critic
        if self.goal_to_critic:
            self.goal_size = corpus.goal_size
            self.input_dim += self.goal_size

        # self.input_dim = self.state_dim + 50 + self.action_dim

        self.use_gpu = config.use_gpu

        self.state_encoder = copy.deepcopy(cvae.utt_encoder)
        if self.word_plas:
            self.action_encoder = copy.deepcopy(cvae.aux_encoder)
        else:
            self.action_encoder = None

        # if self.goal_to_critic:
            # self.q11 = nn.Linear(self.input_dim, 500)
            # self.q12 = nn.Linear(500, 1)
        # else:
        self.q11 = nn.Linear(self.input_dim, 1)
        self.activation_function = args.critic_actf if "critic_actf" in args else "none"

        self.critic_dropout = args.critic_dropout
        if self.critic_dropout:
            self.d = th.nn.Dropout(p=args.critic_dropout_rate, inplace=False)
        else:
            self.d = th.nn.Dropout(p=0.0, inplace=False)

    def forward(self, data_feed, act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, _, _ = self.state_encoder(ctx.unsqueeze(1))
        if self.word_plas:
            resp_summary, _, _ = self.action_encoder(act.unsqueeze(1))
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, resp_summary.squeeze(1)], dim=1)
        else:
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, act], dim=1)

        if self.goal_to_critic:
            try:
                goals = np2var(data_feed['goals'], FLOAT, self.use_gpu)
            except KeyError:
                goals = []
                for turn_id in range(len(ctx_summary)):
                    goals.append(np.concatenate([data_feed['goals_list'][d][turn_id] for d in range(7)]))
                goals = np2var(np.asarray(goals), FLOAT, self.use_gpu)
            sa = th.cat([sa, goals], dim = 1)

        # metadata_summary = self.metadata_encoder(th.cat([bs_label, db_label], dim=1))
        # sa = th.cat([ctx_summary.squeeze(1), metadata_summary, act], dim=1)
        # if self.is_gauss:
            # q1 = F.relu(self.q11(self.d(sa)))
        # else:
        # q1 = F.sigmoid(self.q11(self.d(sa)))
        q1 = self.q11(self.d(sa))
        # if self.goal_to_critic:
            # q1 = self.q12(q1)

        if self.activation_function == "relu":
            q1 = F.relu(q1)
        elif self.activation_function == "sigmoid":
            q1 = F.sigmoid(q1)

        return q1

class SingleHierarchicalRecurrentCritic(nn.Module):
    def __init__(self, cvae, corpus, config, args):
        super(SingleHierarchicalRecurrentCritic, self).__init__()

        # self.vocab = corpus.vocab
        # self.vocab_dict = corpus.vocab_dict
        # self.vocab_size = len(self.vocab)
        # self.bos_id = self.vocab_dict[BOS]
        # self.eos_id = self.vocab_dict[EOS]
        # self.pad_id = self.vocab_dict[PAD]

        self.hidden_size = 500

        if "gauss" in args.sv_config_path:
            self.is_gauss = True
        else:
            self.is_gauss = False
        self.embedding = None
        self.word_plas = args.word_plas
        self.state_dim = cvae.utt_encoder.output_size
        if self.word_plas:
            self.action_dim = cvae.aux_encoder.output_size
        else:
            if self.is_gauss:
                self.action_dim = config.y_size 
            else:
                if args.embed_z_for_critic:
                    self.action_dim = config.dec_cell_size
                else:
                    self.action_dim = config.y_size * config.k_size

        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.input_dim = self.state_dim + self.bs_size + self.db_size + self.action_dim
        # self.input_dim = self.state_dim + 50 + self.action_dim

        self.goal_to_critic = args.goal_to_critic
        self.add_goal = args.add_goal
        if self.goal_to_critic:
            self.goal_size = corpus.goal_size
            if self.add_goal == "early":
                self.input_dim += self.goal_size


        self.use_gpu = config.use_gpu

        self.state_encoder = copy.deepcopy(cvae.utt_encoder)
        if self.word_plas:
            self.action_encoder = copy.deepcopy(cvae.aux_encoder)
        else:
            self.action_encoder = None

        self.dialogue_encoder = nn.LSTM(
                input_size = self.input_dim,
                hidden_size = self.hidden_size,
                dropout=0.1
                )

        if self.add_goal=="late":
            self.q11 = nn.Linear(self.hidden_size + self.goal_size, 1)
        else:
            self.q11 = nn.Linear(self.hidden_size, 1)
        self.activation_function = args.critic_actf if "critic_actf" in args else "none"

        self.critic_dropout = args.critic_dropout
        if self.critic_dropout:
            self.d = th.nn.Dropout(p=args.critic_dropout_rate, inplace=False)
        else:
            self.d = th.nn.Dropout(p=0.0, inplace=False)

        if args.critic_actf == "tanh" or args.critic_actf == "sigmoid":
            self.maxq = args.critic_maxq
        else:
            self.maxq = None

    def forward(self, data_feed, act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, _, _ = self.state_encoder(ctx.unsqueeze(1))
        if self.word_plas:
            resp_summary, _, _ = self.action_encoder(act.unsqueeze(1))
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, resp_summary.squeeze(1)], dim=1)
        else:
            sa = th.cat([ctx_summary.squeeze(1), bs_label, db_label, act], dim=1)

        if self.goal_to_critic:
            try:
                goals = np2var(data_feed['goals'], FLOAT, self.use_gpu)
            except KeyError:
                goals = []
                for turn_id in range(len(ctx_summary)):
                    goals.append(np.concatenate([data_feed['goals_list'][d][turn_id] for d in range(7)]))
                goals = np2var(np.asarray(goals), FLOAT, self.use_gpu)

        #OPTION 1 add goal to encoder for each time step
        if self.goal_to_critic and self.add_goal=="early":
            sa = th.cat([sa, goals], dim = 1)

        output, (hn, cn) = self.dialogue_encoder(self.d(sa.unsqueeze(1)))

        #OPTION 2 add goal combined with hidden state to predict final score
        if self.goal_to_critic and self.add_goal=="late":
            output = th.cat([output, goals.unsqueeze(1)], dim = 2)

        q1 = self.q11(output.squeeze(1))

        if self.activation_function == "relu":
            q1 = F.relu(q1)
        elif self.activation_function == "sigmoid":
            q1 = th.sigmoid(q1)
        elif self.activation_function == "tanh":
            q1 = F.tanh(q1)

        if self.maxq is not None:
            q1 *= self.maxq

        return q1
    
    def forward_target(self, data_feed, act, corpus_act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, _, _ = self.state_encoder(ctx.unsqueeze(1))
        q1s =[]
        for i in range(bs_label.shape[0]):
            if self.word_plas:
                corpus_resp_summary, _, _ = self.action_encoder(corpus_act[:-i].unsqueeze(1))
                actor_resp_summary, _, _ = self.action_encoder(act[i].unsqueeze(1))
                sa = th.cat([ctx_summary[:i+1].squeeze(1), bs_label[:i+1], db_label[:i+1], th.cat([corpus_resp_summary[:i], actor_resp_summary[i]], dim=0).squeeze(1)], dim=1)
            else:
                sa = th.cat([ctx_summary[:i+1].squeeze(1), bs_label[:i+1], db_label[:i+1], th.cat([corpus_act[:i], act[i].unsqueeze(0)], dim=0)], dim=1)

            if self.goal_to_critic:
                try:
                    goals = np2var(data_feed['goals'][:i+1], FLOAT, self.use_gpu)
                except KeyError:
                    goals = []
                    for turn_id in range(i+1):
                        goals.append(np.concatenate([data_feed['goals_list'][d][turn_id] for d in range(7)]))
                    goals = np2var(np.asarray(goals), FLOAT, self.use_gpu)

            #OPTION 1 add goal to encoder for each time step
            if self.goal_to_critic and self.add_goal=="early":
                sa = th.cat([sa, goals], dim = 1)

            output, (hn, cn) = self.dialogue_encoder(self.d(sa.unsqueeze(1)))

            #OPTION 2 add goal combined with hidden state to predict final score
            if self.goal_to_critic and self.add_goal=="late":
                output = th.cat([output, goals.unsqueeze(1)], dim = 2)

            q1 = self.q11(output.squeeze(1))

            if self.activation_function == "relu":
                q1 = F.relu(q1)
            elif self.activation_function == "sigmoid":
                q1 = F.sigmoid(q1)
            elif self.activation_function == "tanh":
                q1 = F.tanh(q1) * self.maxq

            q1s.append(q1[-1])

        return th.cat(q1s, dim=0).unsqueeze(1)

class SingleTransformersCritic(nn.Module):
    def __init__(self, cvae, corpus, config, args):
        super(SingleTransformersCritic, self).__init__()

        # self.vocab = corpus.vocab
        # self.vocab_dict = corpus.vocab_dict
        # self.vocab_size = len(self.vocab)
        # self.bos_id = self.vocab_dict[BOS]
        # self.eos_id = self.vocab_dict[EOS]
        # self.pad_id = self.vocab_dict[PAD]

        self.hidden_size = 128

        if "gauss" in args.sv_config_path:
            self.is_gauss = True
        else:
            self.is_gauss = False
        self.embedding = None
        self.word_plas = args.word_plas
        self.state_dim = cvae.utt_encoder.output_size
        if self.word_plas:
            self.action_dim = cvae.aux_encoder.output_size
        else:
            if self.is_gauss:
                self.action_dim = config.y_size 
            else:
                if args.embed_z_for_critic:
                    self.action_dim = config.dec_cell_size
                else:
                    self.action_dim = config.y_size * config.k_size

        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.input_dim = self.state_dim + self.bs_size + self.db_size + self.action_dim
        self.db_embedding = nn.Linear(self.db_size, config.embed_size)
        self.bs_embedding = nn.Linear(self.bs_size, config.embed_size)

        self.goal_to_critic = args.goal_to_critic
        if self.goal_to_critic:
            raise NotImplementedError


        self.use_gpu = config.use_gpu

        self.state_encoder = copy.deepcopy(cvae.utt_encoder)
        if self.word_plas:
            self.action_encoder = copy.deepcopy(cvae.aux_encoder)
        else:
            self.action_encoder = None

        self.trans_encoder_layer = nn.TransformerEncoderLayer(nhead=8, d_model=config.embed_size)
        self.trans_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=4)

        self.dialogue_encoder = nn.LSTM(
                input_size = config.embed_size,
                hidden_size = self.hidden_size,
                dropout=0.1
                )
        if not self.word_plas:
            self.act_embedding = nn.Linear(self.action_dim, config.embed_size)
        self.bs_encoder = nn.Linear(self.db_size, config.embed_size)
        self.db_encoder = nn.Linear(self.db_size, config.embed_size)

        self.q11 = nn.Linear(self.hidden_size, 1)

        self.critic_dropout = args.critic_dropout
        if self.critic_dropout:
            self.d = th.nn.Dropout(p=args.critic_dropout_rate, inplace=False)
        else:
            self.d = th.nn.Dropout(p=0.0, inplace=False)

    def forward(self, data_feed, act):
        ctx = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        ctx_summary, word_emb, enc_outs = self.state_encoder(ctx.unsqueeze(1))
        # word_emb : (batch_size, max_len, 256)
        # enc_outs : (batch_size, max_len, 600)
        metadata_embedding = th.cat([self.bs_embedding(bs_label).unsqueeze(1), self.db_embedding(db_label).unsqueeze(1)], dim=1)

        if self.word_plas:
            resp_summary, resp_word_emb, resp_enc_outs = self.action_encoder(act.unsqueeze(1))
            act_embedding = resp_word_emb
        else:
            act_embedding = self.act_embedding(act).unsqueeze(1)

        sa = th.cat([word_emb, metadata_embedding, act_embedding], dim=1)
        sa = self.trans_encoder(self.d(sa))
        output, (hn, cn) = self.dialogue_encoder(self.d(sa))
        q1 = F.sigmoid(self.q11(output[:, -1].squeeze(1)))
        # q1 = self.q11(q1[:, 0])


        return q1


class CatActor(nn.Module):
    def __init__(self, model, corpus, config):
        super(CatActor, self).__init__()
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.config = config

        self.use_gpu = config.use_gpu

        self.embedding = None
        self.y_size = config.y_size
        self.k_size = config.k_size
        # self.max_action = config.max_action
        self.is_gauss = False
        self.is_stochastic = config.is_stochastic

        self.utt_encoder = copy.deepcopy(model.utt_encoder)

        self.policy = copy.deepcopy(model.c2z)
        if self.is_stochastic:
            self.gumbel_connector = copy.deepcopy(model.gumbel_connector)

    def forward(self, data_feed):
        short_ctx_utts = np2var(extract_short_ctx(data_feed['contexts'], data_feed['context_lens']), LONG, self.use_gpu)
        bs_label = np2var(data_feed['bs'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = np2var(data_feed['db'], FLOAT, self.use_gpu)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)


        logits_qy, log_qy = self.policy(enc_last)
        if self.is_stochastic:
            z = self.gumbel_connector(logits_qy, hard=True)
            soft_z = self.gumbel_connector(logits_qy, hard=False)
        else:
            z_idx = th.argmax(th.exp(log_qy), dim=1, keepdim=True)
            z = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            z.scatter_(1, z_idx, 1.0)
            soft_z = th.exp(log_qy)

        return z, soft_z, log_qy



class ReplayBuffer(object):
    """
    Buffer to store experiences, to be used in off-policy learning
    """
    def __init__(self, config): 
        # true_responses = id2sent(next_state)
        # pred_responses = model.z2x(action)

        self.batch_size = config.batch_size
        self.fix_episode = config.fix_episode
        
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "next_action", "done", "Return"])
        self.memory = deque()
        self.seed = random.seed(config.random_seed)
        # self.reinforce_data = config.reinforce_data

    def add(self, states, actions, rewards, next_states, next_actions, dones, Returns):
        if self.fix_episode:
            self._add_episode(states, actions, rewards, next_states, next_actions, dones, Returns)
        else:
            for i in range(len(states)):
                self._add(states[i], actions[i], rewards[i], next_states[i], next_actions[i], dones[i], Returns[i])

    def _add(self, state, action, reward, next_state, next_action, done, Return):
        e = self.experiences(state, action, reward, next_state, next_action, done, Return)
        self.memory.append(e)

    def _add_episode(self, states, actions, rewards, next_states, next_actions, dones, Returns):
        ep = []
        for s, a, r, s_, a_, d, R in zip(states, actions, rewards, next_states, next_actions, dones, Returns):
            ep.append(self.experiences(s, a, r, s_, a_, d, R))
        self.memory.append(ep)


    def sample(self):
        if self.fix_episode:
            return self._sample_episode()
        else:
            return self._sample()

    def _sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)

        states = {}
        states['contexts'] = np.asarray([e.state['contexts'] for e in experiences])
        states['bs'] = np.asarray([e.state['bs'] for e in experiences])
        states['db'] = np.asarray([e.state['db'] for e in experiences])
        states['context_lens'] = np.asarray([e.state['context_lens'] for e in experiences]) 
        states['goals'] = np.asarray([e.state['goals'] for e in experiences]) 

        actions = np.asarray([e.action for e in experiences if e is not None])
        rewards = np.asarray([e.reward for e in experiences if e is not None])
        
        next_states = {}
        next_states['contexts'] = np.asarray([e.next_state['contexts'] for e in experiences])
        next_states['bs'] = np.asarray([e.next_state['bs'] for e in experiences])
        next_states['db'] = np.asarray([e.next_state['db'] for e in experiences])
        next_states['context_lens'] = np.asarray([e.next_state['context_lens'] for e in experiences]) 
        next_states['goals'] = np.asarray([e.next_state['goals'] for e in experiences])
        
        next_actions = np.asarray([e.next_action for e in experiences if e is not None])

        dones = np.asarray([e.done for e in experiences if e is not None])
        returns = np.asarray([e.Return for e in experiences if e is not None])
        # if self.reinforce_data:
            # rewards = dones * 10 + 1 # give positive rewards to all actions taken in the data

        return (states, actions, rewards, next_states, next_actions, dones, returns)
        # return experiences
    
    def _sample_episode(self):
        # episodes = random.sample(self.memory, k = self.batch_size)
        episodes = random.sample(self.memory, k = 1)

        for experiences in episodes:
            states = {}
            states['contexts'] = np.asarray([e.state['contexts'] for e in experiences])
            states['bs'] = np.asarray([e.state['bs'] for e in experiences])
            states['db'] = np.asarray([e.state['db'] for e in experiences])
            states['keys'] = [e.state['keys'] for e in experiences]
            states['context_lens'] = np.asarray([e.state['context_lens'] for e in experiences]) 
            states['goals'] = np.asarray([e.state['goals'] for e in experiences]) 

            actions = np.asarray([e.action for e in experiences if e is not None])
            rewards = np.asarray([e.reward for e in experiences if e is not None])
            
            next_states = {}
            next_states['contexts'] = np.asarray([e.next_state['contexts'] for e in experiences])
            next_states['bs'] = np.asarray([e.next_state['bs'] for e in experiences])
            next_states['db'] = np.asarray([e.next_state['db'] for e in experiences])
            next_states['keys'] = [e.next_state['keys'] for e in experiences]
            next_states['context_lens'] = np.asarray([e.next_state['context_lens'] for e in experiences]) 
            next_states['goals'] = np.asarray([e.next_state['goals'] for e in experiences])
            
            next_actions = np.asarray([e.next_action for e in experiences if e is not None])

            dones = np.asarray([e.done for e in experiences if e is not None])
            returns = np.asarray([e.Return for e in experiences if e is not None])
            # if self.reinforce_data:
                # rewards = dones * 10 + 1 # give positive rewards to all actions taken in the data

        return (states, actions, rewards, next_states, next_actions, dones, returns)
        # return experiences

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.memory, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = dill.load(f)
