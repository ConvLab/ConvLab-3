import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from convlab2.policy.lava.multiwoz.latent_dialog.base_models import BaseModel, frange_cycle_linear
from convlab2.policy.lava.multiwoz.latent_dialog.corpora import SYS, EOS, PAD, BOS, DOMAIN_REQ_TOKEN, ACTIVE_BS_IDX, NO_MATCH_DB_IDX, REQ_TOKENS
from convlab2.policy.lava.multiwoz.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from convlab2.policy.lava.multiwoz.latent_dialog.enc2dec.encoders import RnnUttEncoder
from convlab2.policy.lava.multiwoz.latent_dialog.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from convlab2.policy.lava.multiwoz.latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss, GaussianEntropy
from convlab2.policy.lava.multiwoz.latent_dialog import nn_lib
import numpy as np
import pdb
import json


class SysPerfectBD2Word(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Word, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.policy = nn.Sequential(nn.Linear(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                              config.dec_cell_size), nn.Tanh(), nn.Dropout(config.dropout))

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.utt_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            # h_dec_init_state = utt_summary.squeeze(1).unsqueeze(0)
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=self.nll(dec_outputs, labels),
                        latent_action=dec_init_state)
        else:
            return Pack(nll=self.nll(dec_outputs, labels))

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=temp)
        return logprobs, outs

class SysPerfectBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        if "policy_dropout" in config and config.policy_dropout:
            if "policy_dropout_rate" in config:
                self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                  config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
            else:
                self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                  config.y_size, config.k_size, is_lstm=False)

        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                              config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        if "state_for_decoding" not in self.config:
            self.state_for_decoding = False
        else:
            self.state_for_decoding = self.config.state_for_decoding

        if self.state_for_decoding:
            dec_hidden_size = config.dec_cell_size + self.utt_encoder.output_size + self.db_size + self.bs_size
        else:
            dec_hidden_size = config.dec_cell_size


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)

        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        if "kl_annealing" in self.config and config.kl_annealing=="cyclical":
            self.beta = frange_cycle_linear(config.n_iter, start=self.config.beta_start, stop=self.config.beta_end, n_cycle=10)    
        else:
            self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0

        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if isinstance(self.beta, float):
            beta = self.beta
        else:
            if batch_cnt == None:
                beta = self.beta[-1]
            else:
                beta = self.beta[int(batch_cnt)]


        if self.simple_posterior or "kl_annealing" in self.config:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last) # p(z|c)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=True)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=False)
        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15) # averaged over all samples
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        # if np.random.rand() < epsilon: # greedy exploration
            # print("randomly sampling latent")
            # idx = th.multinomial(th.cuda.FloatTensor(qy.shape).uniform_(), 1)
        # else: # normal latent sampling
        idx = th.multinomial(qy, 1).detach()
        
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y
    
    def sample_z(self, data_feed, n_z=1, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        zs = []
        logpzs = []
        for i in range(n_z):
            idx = th.multinomial(qy, 1).detach()
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
            joint_logpz = th.sum(logprob_sample_z, dim=1)
            sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

            zs.append(sample_y)
            logpzs.append(joint_logpz)

        
        return th.stack(zs), th.stack(logpzs)
    
    def sample_z_with_exploration(self, data_feed, n_z=1, temp=0.1, epsilon=0.05):
        #TODO consider deleting this function
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        zs = []
        logpzs = []
        for i in range(n_z):
            if np.random.rand() < epsilon: # greedy exploration
                idx = th.multinomial(th.cuda.FloatTensor(qy.shape).uniform_(), 1)
            else: # normal latent sampling
                idx = th.multinomial(qy, 1).detach()
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
            joint_logpz = th.sum(logprob_sample_z, dim=1)
            sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

            zs.append(sample_y)
            logpzs.append(joint_logpz)

        
        return th.stack(zs), th.stack(logpzs)

    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        
        if data_feed:
            ctx_lens = data_feed['context_lens']  # (batch_size, )
            short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
 
        # pack attention context
        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        if self.config.dec_use_attn:
           z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
           attn_context = []
           temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
           for z_id in range(self.y_size):
               attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
           attn_context = th.cat(attn_context, dim=1)
           dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
           dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
           attn_context = None

        # decode
        if self.state_for_decoding:
            utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # create decoder initial states
            enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)


        #dec_init_state = self.np2var(dec_init_state, FLOAT).unsqueeze(0)
        #attn_context = self.np2var(attn_context, FLOAT)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # has to be forward_rl because we don't have the golden target
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                temp=temp)
        return logprobs, outs

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class SysEncodedBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysEncodedBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.config = config
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        if config.use_metadata_for_decoding:
            self.metadata_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                             embedding_dim=int(config.embed_size / 2),
                                             feat_size=0,
                                             goal_nhid=0,
                                             rnn_cell=config.utt_rnn_cell,
                                             utt_cell_size=int(config.dec_cell_size / 2),
                                             num_layers=config.num_layers,
                                             input_dropout_p=config.dropout,
                                             output_dropout_p=config.dropout,
                                             bidirectional=config.bi_utt_cell,
                                             variable_lengths=False,
                                             use_attn=config.enc_use_attn,
                                             embedding=self.embedding)

        if "policy_dropout" in config and config.policy_dropout:
            if "policy_dropout_rate" in config:
                self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size,
                                  config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
            else:
                self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size,
                                  config.y_size, config.k_size, is_lstm=False)

        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        if "state_for_decoding" not in self.config:
            self.state_for_decoding = False
        else:
            self.state_for_decoding = self.config.state_for_decoding

        dec_hidden_size = config.dec_cell_size
        if config.use_metadata_for_decoding:
            if "metadata_to_decoder" not in config or config.metadata_to_decoder == "concat":
                dec_hidden_size += self.metadata_encoder.output_size
        if self.state_for_decoding:
            dec_hidden_size += self.utt_encoder.output_size

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)

        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))

        if "kl_annealing" in self.config and config.kl_annealing=="cyclical":
            self.beta = frange_cycle_linear(config.n_iter, start=self.config.beta_start, stop=self.config.beta_end, n_cycle=10)    
        else:
            self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0


        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if isinstance(self.beta, float):
            beta = self.beta
        else:
            if batch_cnt == None:
                beta = self.beta[-1]
            else:
                beta = self.beta[int(batch_cnt % self.config.n_iter)]
               
        if self.simple_posterior or "kl_annealing" in self.config:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def extract_short_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['contexts']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            for t_id in range(ctx_lens[b_id]):
                utt.extend(context[b_id][t_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)
    
    def extract_metadata(self, data_feed):
        utts = []
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(bs)):
            utt = []
            if "metadata_db_only" in self.config and self.config.metadata_db_only:
                utt.extend(db[b_id])
            else:
                utt.extend(bs[b_id] + db[b_id])
            utts.append(self.pad_to(self.config.max_metadata_len, utt, do_pad=True))
        return np.array(utts)

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        enc_last = utt_summary.unsqueeze(1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last.squeeze(), x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=True)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=False)
        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        
        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)

        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, th.transpose(enc_last.squeeze(1), 1, 0)], dim=2)
        
        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,   # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15) # averaged over all samples
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        enc_last = utt_summary.unsqueeze(1)
        # create decoder initial states
        logits_py, log_qy = self.c2z(enc_last)
        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)
        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        
        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
        
        # decode
        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, th.transpose(enc_last.squeeze(1), 1, 0)], dim=2)
 
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y
    
    def sample_z(self, data_feed, n_z=1, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        # metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
        # out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        # metadata_summary, _, metadata_enc_outs = self.utt_encoder(metadata.unsqueeze(1))


        # create decoder initial states
        enc_last = utt_summary.unsqueeze(1)
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        zs = []
        logpzs = []
        for i in range(n_z):
            idx = th.multinomial(qy, 1).detach()
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
            joint_logpz = th.sum(logprob_sample_z, dim=1)
            sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

            zs.append(sample_y)
            logpzs.append(joint_logpz)

        
        return th.stack(zs), th.stack(logpzs)

    def decode_z(self, sample_y, batch_size, max_words=None, temp=1.0, gen_type='greedy'):
        """
        generate response from latent var
        """
        # pack attention context
        metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
        metadata_summary, _, metadata_enc_outs = self.utt_encoder(metadata.unsqueeze(1))

        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        if self.config.dec_use_attn:
           z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
           attn_context = []
           temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
           for z_id in range(self.y_size):
               attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
           attn_context = th.cat(attn_context, dim=1)
           dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
           dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
           attn_context = None

        dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)

        if self.config.use_metadata_for_decoding:
            raise NotImplementedError

        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, th.transpose(enc_last.squeeze(1), 1, 0)], dim=2)
 
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # has to be forward_rl because we don't have the golden target
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                temp=temp)
        return logprobs, outs

class SysAECat(BaseModel):
    def __init__(self, corpus, config):
        super(SysAECat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = True # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = False # does not use context cause AE task

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        
        if "ae_zero_padding" in self.config and self.config.ae_zero_padding:
            # self.use_metadata = self.config.use_metadata
            self.ae_zero_padding = self.config.ae_zero_padding
            c2z_input_size = self.utt_encoder.output_size + self.db_size + self.bs_size
        else:
            # self.use_metadata = False
            self.ae_zero_padding = False
            c2z_input_size = self.utt_encoder.output_size


        if "policy_dropout" in config and config.policy_dropout:
            self.c2z = nn_lib.Hidden2DiscretewDropout(c2z_input_size,
                                              config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
        else:
            self.c2z = nn_lib.Hidden2Discrete(c2z_input_size,
                                              config.y_size, config.k_size, is_lstm=False)

        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        # if not self.simple_posterior: #q(z|x,c)
            # if self.contextual_posterior:
                # # x, c, BS, and DB
                # self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   # config.y_size, config.k_size, is_lstm=False)
            # else:
                # self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)


        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)
        
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        if self.ae_zero_padding:
            enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), utt_summary.squeeze(1)], dim=1)
        else:
            enc_last = utt_summary.squeeze(1)


        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        # else:
            # logits_py, log_py = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or (use_py is not None and use_py is True):
                # sample_y = self.gumbel_connector(logits_py, hard=False)
            # else:
                # sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y

class SysGroundedAECat(BaseModel):
    def __init__(self, corpus, config):
        super(SysGroundedAECat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = True # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = False # does not use context cause AE task

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        if config.use_metadata_for_decoding:
            self.metadata_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                             embedding_dim=int(config.embed_size / 2),
                                             feat_size=0,
                                             goal_nhid=0,
                                             rnn_cell=config.utt_rnn_cell,
                                             utt_cell_size=int(config.dec_cell_size / 2),
                                             num_layers=config.num_layers,
                                             input_dropout_p=config.dropout,
                                             output_dropout_p=config.dropout,
                                             bidirectional=config.bi_utt_cell,
                                             variable_lengths=False,
                                             use_attn=config.enc_use_attn,
                                             embedding=self.embedding)

        if "policy_dropout" in config and config.policy_dropout:
            self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        # if not self.simple_posterior: #q(z|x,c) # use bs and db grounding as c
            # if self.contextual_posterior:
                # x, c, BS, and DB
                # self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   # config.y_size, config.k_size, is_lstm=False)
            # else:
            # self.xc2z = nn_lib.Hidden2Discrete(self.metadata_encoder.output_size, config.y_size, config.k_size, is_lstm=False) # prior network conditioned on BS+DB

        if config.use_metadata_for_decoding:
            if "metadata_to_decoder" not in config or config.metadata_to_decoder == "concat":
                dec_hidden_size = config.dec_cell_size + self.metadata_encoder.output_size
            else:
                dec_hidden_size = config.dec_cell_size
        else:
            dec_hidden_size = config.dec_cell_size



        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)


        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off {} to {}".format(len(tokens), max_len))
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def extract_AE_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['outputs']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            utt.extend(context[b_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def extract_metadata(self, data_feed):
        utts = []
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(bs)):
            utt = []
            if "metadata_db_only" in self.config and self.config.metadata_db_only:
                utt.extend(db[b_id])
            else:
                utt.extend(bs[b_id] + db[b_id])
            utts.append(self.pad_to(self.config.max_metadata_len, utt, do_pad=True))
        return np.array(utts)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        if self.config.use_metadata_for_encoder:
            ctx_utts = self.np2var(self.extract_AE_ctx(data_feed), LONG) # contains bs and db
            utt_summary, _, enc_outs = self.utt_encoder(ctx_utts.unsqueeze(1))
        else:
            in_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
            utt_summary, _, enc_outs = self.utt_encoder(in_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.squeeze(1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        # else:
            # logits_py, log_py = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or (use_py is not None and use_py is True):
                # sample_y = self.gumbel_connector(logits_py, hard=False)
            # else:
                # sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

class SysGroundedAEGauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysGroundedAEGauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.y_size = config.y_size
        self.simple_posterior = True # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = False # does not use context cause AE task

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        self.metadata_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=int(config.embed_size / 2),
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=int(config.utt_cell_size / 2),
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        if "policy_dropout" in config and config.policy_dropout:
            raise NotImplementedError
        else:
            self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size,
                                          config.y_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size, bias=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        if not self.simple_posterior:
            # self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               # config.y_size, is_lstm=False)
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size + self.metadata_encoder.output_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)

        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def extract_AE_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['outputs']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            utt.extend(context[b_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def extract_metadata(self, data_feed):
        utts = []
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(bs)):
            utt = []
            utt.extend(bs[b_id] + db[b_id])
            utts.append(self.pad_to(self.config.max_metadata_len, utt, do_pad=True))
        return np.array(utts)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
        batch_size = len(ctx_lens)

        if self.config.use_metadata_for_encoder:
            ctx_utts = self.np2var(self.extract_AE_ctx(data_feed), LONG) # contains bs and db
            utt_summary, _, enc_outs = self.utt_encoder(ctx_utts.unsqueeze(1))
        else:
            in_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
            utt_summary, _, enc_outs = self.utt_encoder(in_utts.unsqueeze(1))
        metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.squeeze(1)
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        # else:
            # p_mu, p_logvar = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or use_py:
                # sample_z = self.gauss_connector(p_mu, p_logvar)
            # else:
                # sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None
        # decode
        dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result
    
    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

class SysMTCat(BaseModel):
    def __init__(self, corpus, config): 
        super(SysMTCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = config.contextual_posterior # does not use context cause AE task
        self.shared_train = config.shared_train

        if "use_aux_kl" in config:
            self.use_aux_kl = config.use_aux_kl
        else:
            self.use_aux_kl = False

        self.embedding = None
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)


        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        
        if not self.simple_posterior: #q(z|x,c)
            if self.contextual_posterior:
                # x, c, BS, and DB
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if "aux_pi_beta" in self.config:
            self.aux_pi_beta = self.config.aux_pi_beta
        else:
            self.aux_pi_beta = 1.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.shared_train:
            if "selective_fine_tune" in self.config and self.config.selective_fine_tune:
                total_loss = loss.nll + self.config.beta * loss.aux_pi_kl
            else:
                total_loss = loss.nll + loss.ae_nll + self.config.aux_pi_beta * loss.aux_pi_kl + self.config.beta * loss.aux_kl 
        else:
            if self.simple_posterior:
                total_loss = loss.nll
                if self.config.use_pr > 0.0:
                    total_loss += self.config.beta * loss.pi_kl
            else:
                total_loss = loss.nll + loss.pi_kl


        return total_loss
    
    def encode_state(self, data_feed):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        return enc_last

    def encode_action(self, data_feed):
        batch_size = data_feed.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(data_feed.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = aux_utt_summary.squeeze(1)

        return aux_enc_last

    def get_z_via_vae(self, data_feed, hard=False):
        batch_size = data_feed.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(data_feed.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = th.cat([self.np2var(np.zeros([batch_size, self.bs_size]), LONG), self.np2var(np.zeros([batch_size, self.db_size]), LONG), aux_utt_summary.squeeze(1)], dim=1)

        logits_qy, log_qy = self.c2z(aux_enc_last)
        aux_sample_z = self.gumbel_connector(logits_qy, hard=hard)
        
        return aux_sample_z

    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        
        # if data_feed:
            # ctx_lens = data_feed['context_lens']  # (batch_size, )
            # short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            # bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            # db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
 
        # pack attention context
        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        if self.config.dec_use_attn:
           z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
           attn_context = []
           temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
           for z_id in range(self.y_size):
               attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
           attn_context = th.cat(attn_context, dim=1)
           dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
           dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
           attn_context = None

        # decode
        # if self.state_for_decoding:
            # utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # # create decoder initial states
            # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

            # dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)


        #dec_init_state = self.np2var(dec_init_state, FLOAT).unsqueeze(0)
        #attn_context = self.np2var(attn_context, FLOAT)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # has to be forward_rl because we don't have the golden target
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                temp=temp)
        return logprobs, outs

    def forward_aux(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.aux_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        
        # how to use z, alone or in combination with bs and db
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=False)
            sample_y_discrete = self.gumbel_connector(logits_qy, hard=True)
            log_py = self.log_uniform_y
        # else:
            # logits_py, log_py = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or (use_py is not None and use_py is True):
                # sample_y = self.gumbel_connector(logits_py, hard=False)
            # else:
                # sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        short_target_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        aux_enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), aux_utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            if self.shared_train:
                aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
                aux_sample_y = self.gumbel_connector(aux_logits_qy, hard=mode==GEN)

            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])


        if self.shared_train:
            if self.config.dec_use_attn:
                aux_z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
                aux_attn_context = []
                aux_temp_sample_y = aux_sample_y.view(-1, self.config.y_size, self.config.k_size)
                for z_id in range(self.y_size):
                    aux_attn_context.append(th.mm(aux_temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
                aux_attn_context = th.cat(aux_attn_context, dim=1)
                aux_dec_init_state = th.sum(aux_attn_context, dim=1).unsqueeze(0)
            else:
                aux_dec_init_state = self.z_embedding(aux_sample_y.view(1, -1, self.config.y_size * self.config.k_size))
                aux_attn_context = None
            if self.config.dec_rnn_cell == 'lstm':
                aux_dec_init_state = tuple([aux_dec_init_state, aux_dec_init_state])



        # decode
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            if self.shared_train:
                ae_dec_outputs, ae_dec_hidden_state, ae_ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=aux_dec_init_state,  # tuple: (h, c)
                                                               attn_context=aux_attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
                result['ae_nll'] = self.nll(ae_dec_outputs, labels)
                aux_pi_kl = self.cat_kl_loss(log_qy, aux_log_qy, batch_size, unit_average=True)
                aux_kl = self.cat_kl_loss(aux_log_qy, log_py, batch_size, unit_average=True)
                result['aux_pi_kl'] = aux_pi_kl
                result['aux_kl'] = aux_kl


            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens
    
class SysGroundedMTCat(BaseModel):
    def __init__(self, corpus, config): 
        super(SysGroundedMTCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = config.contextual_posterior # does not use context cause AE task

        if "use_aux_kl" in config:
            self.use_aux_kl = config.use_aux_kl
        else:
            self.use_aux_kl = False

        self.embedding = None
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                        embedding=self.embedding)

        if config.use_metadata_for_decoding:
            self.metadata_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                             embedding_dim=int(config.embed_size / 2),
                                             feat_size=0,
                                             goal_nhid=0,
                                             rnn_cell=config.utt_rnn_cell,
                                             utt_cell_size=int(config.dec_cell_size / 2),
                                             num_layers=config.num_layers,
                                             input_dropout_p=config.dropout,
                                             output_dropout_p=config.dropout,
                                             bidirectional=config.bi_utt_cell,
                                             variable_lengths=False,
                                             use_attn=config.enc_use_attn,
                                             embedding=self.embedding)



        if "policy_dropout" in config and config.policy_dropout:
            self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False)


        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        
        if not self.simple_posterior: #q(z|x,c)
            if self.contextual_posterior:
                # x, c, BS, and DB
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)
        
        if config.use_metadata_for_decoding:
            if "metadata_to_decoder" not in config or config.metadata_to_decoder == "concat":
                dec_hidden_size = config.dec_cell_size + self.metadata_encoder.output_size
            else:
                dec_hidden_size = config.dec_cell_size
        else:
            dec_hidden_size = config.dec_cell_size


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        if self.use_aux_kl:
            try:
                total_loss += loss.aux_pi_kl
            except KeyError:
                total_loss += 0

        return total_loss

    def forward_aux(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):

        ctx_lens = data_feed['context_lens']  # (batch_size, )
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        if self.config.use_metadata_for_aux_encoder:
            ctx_outs = self.np2var(self.extract_AE_ctx(data_feed), LONG) # contains bs and db
            utt_summary, _, _ = self.aux_encoder(ctx_outs.unsqueeze(1))
        else:
            short_target_utts = self.np2var(data_feed['outputs'], LONG)
            utt_summary, _, _ = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.unsqueeze(1)
        
        # how to use z, alone or in combination with bs and db
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        # else:
            # logits_py, log_py = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or (use_py is not None and use_py is True):
                # sample_y = self.gumbel_connector(logits_py, hard=False)
            # else:
                # sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        
        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)


        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.unsqueeze(1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)

        

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result
    
    def shared_forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        if self.config.use_metadata_for_aux_encoder:
            ctx_outs = self.np2var(self.extract_AE_ctx(data_feed), LONG) # contains bs and db
            aux_utt_summary, _, aux_enc_outs = self.aux_encoder(ctx_outs.unsqueeze(1))
        else:
            short_target_utts = self.np2var(data_feed['outputs'], LONG)
            aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.unsqueeze(1)
        aux_enc_last = aux_utt_summary.unsqueeze(1)

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        
        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)


        
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            aux_pi_kl = self.cat_kl_loss(log_qy, aux_log_qy, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['aux_pi_kl'] = aux_pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def extract_metadata(self, data_feed):
        utts = []
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(bs)):
            utt = []
            if "metadata_db_only" in self.config and self.config.metadata_db_only:
                utt.extend(db[b_id])
            else:
                utt.extend(bs[b_id] + db[b_id])
            utts.append(self.pad_to(self.config.max_metadata_len, utt, do_pad=True))
        return np.array(utts)

    def extract_AE_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['outputs']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            utt.extend(context[b_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def extract_short_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['contexts']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            for t_id in range(ctx_lens[b_id]):
                utt.extend(context[b_id][t_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class SysActZCat(BaseModel):
    def __init__(self, corpus, config): 
        super(SysActZCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = config.contextual_posterior # does not use context cause AE task

        if "use_aux_kl" in config:
            self.use_aux_kl = config.use_aux_kl
        else:
            self.use_aux_kl = False
        
        if "use_aux_c2z" in config:
            self.use_aux_c2z = config.use_aux_c2z
        else:
            self.use_aux_c2z = False



        self.embedding = None
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        # if "policy_dropout" in config and config.policy_dropout:
            # self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                              # config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
        # else:
        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        if self.use_aux_c2z:
                self.aux_c2z = nn_lib.Hidden2Discrete(self.aux_encoder.output_size, config.y_size, config.k_size, is_lstm=False)


        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        
        if not self.simple_posterior: #q(z|x,c)
            if self.contextual_posterior:
                # x, c, BS, and DB
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)


        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)

        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss
    
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        short_target_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        aux_enc_last = th.cat([bs_label, db_label, aux_utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            if self.use_aux_c2z:
                aux_logits_qy, aux_log_qy = self.aux_c2z(aux_utt_summary.squeeze(1))
            else:
                aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)

            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = aux_log_qy
        else: 
            logits_py, log_py = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=True)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=False)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def forward_aux(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.aux_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        # else:
            # p_mu, p_logvar = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or use_py:
                # sample_z = self.gauss_connector(p_mu, p_logvar)
            # else:
                # sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None


        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def get_z_via_vae(self, data_feed, hard=False):
        batch_size = data_feed.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(data_feed.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = th.cat([self.np2var(np.zeros([batch_size, self.bs_size]), LONG), self.np2var(np.zeros([batch_size, self.db_size]), LONG), aux_utt_summary.squeeze(1)], dim=1)

        logits_qy, log_qy = self.c2z(aux_enc_last)
        aux_sample_z = self.gumbel_connector(logits_qy, hard=hard)
        
        return aux_sample_z, logits_qy, log_qy
        
    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                  temp=0.1)
        return logprobs, outs, joint_logpz, sample_y
    
    def sample_z(self, data_feed, n_z=1, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        zs = []
        logpzs = []
        for i in range(n_z):
            idx = th.multinomial(qy, 1).detach()
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
            joint_logpz = th.sum(logprob_sample_z, dim=1)
            sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

            zs.append(sample_y)
            logpzs.append(joint_logpz)

        
        return th.stack(zs), th.stack(logpzs)
    
    def sample_z_with_exploration(self, data_feed, n_z=1, temp=0.1, epsilon=0.05):
        #TODO consider deleting this function
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )

        zs = []
        logpzs = []
        for i in range(n_z):
            if np.random.rand() < epsilon: # greedy exploration
                idx = th.multinomial(th.cuda.FloatTensor(qy.shape).uniform_(), 1)
            else: # normal latent sampling
                idx = th.multinomial(qy, 1).detach()
            logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
            joint_logpz = th.sum(logprob_sample_z, dim=1)
            sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

            zs.append(sample_y)
            logpzs.append(joint_logpz)

        
        return th.stack(zs), th.stack(logpzs)

    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        # pack attention context

        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        if self.config.dec_use_attn:
           z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
           attn_context = []
           temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
           for z_id in range(self.y_size):
               attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
           attn_context = th.cat(attn_context, dim=1)
           dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
           dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
           attn_context = None

        
        # decode

        #dec_init_state = self.np2var(dec_init_state, FLOAT).unsqueeze(0)
        #attn_context = self.np2var(attn_context, FLOAT)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # has to be forward_rl because we don't have the golden target
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                temp=temp)

        return logprobs, outs

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class SysGroundedActZCat(BaseModel):
    def __init__(self, corpus, config): 
        super(SysGroundedActZCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = config.contextual_posterior # does not use context cause AE task

        if "use_aux_kl" in config:
            self.use_aux_kl = config.use_aux_kl
        else:
            self.use_aux_kl = False

        self.embedding = None
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        if config.use_metadata_for_decoding:
            self.metadata_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                             embedding_dim=int(config.embed_size / 2),
                                             feat_size=0,
                                             goal_nhid=0,
                                             rnn_cell=config.utt_rnn_cell,
                                             utt_cell_size=int(config.dec_cell_size / 2),
                                             num_layers=config.num_layers,
                                             input_dropout_p=config.dropout,
                                             output_dropout_p=config.dropout,
                                             bidirectional=config.bi_utt_cell,
                                             variable_lengths=False,
                                             use_attn=config.enc_use_attn,
                                             embedding=self.embedding)



        if "policy_dropout" in config and config.policy_dropout:
            self.c2z = nn_lib.Hidden2DiscretewDropout(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False, p_dropout=config.policy_dropout_rate, dropout_on_eval=config.dropout_on_eval)
        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                              config.y_size, config.k_size, is_lstm=False)

        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        
        if not self.simple_posterior: #q(z|x,c)
            if self.contextual_posterior:
                # x, c, BS, and DB
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)
        if config.use_metadata_for_decoding:
            if "metadata_to_decoder" not in config or config.metadata_to_decoder == "concat":
                dec_hidden_size = config.dec_cell_size + self.metadata_encoder.output_size
            else:
                dec_hidden_size = config.dec_cell_size
        else:
            dec_hidden_size = config.dec_cell_size


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)


        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)

        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def extract_short_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['contexts']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            for t_id in range(ctx_lens[b_id]):
                utt.extend(context[b_id][t_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def extract_metadata(self, data_feed):
        utts = []
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(bs)):
            utt = []
            if "metadata_db_only" in self.config and self.config.metadata_db_only:
                utt.extend(db[b_id])
            else:
                utt.extend(bs[b_id] + db[b_id])
            utts.append(self.pad_to(self.config.max_metadata_len, utt, do_pad=True))
        return np.array(utts)

    def extract_AE_ctx(self, data_feed):
        utts = []
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        context = data_feed['outputs']
        bs = data_feed['bs']
        db = data_feed['db']
        if not isinstance(bs, list):
            bs = data_feed['bs'].tolist()
            db = data_feed['db'].tolist()

        for b_id in range(len(context)):
            utt = []
            utt.extend(context[b_id])
            try:
                utt.extend(bs[b_id] + db[b_id])
            except:
                pdb.set_trace()
            utts.append(self.pad_to(self.config.max_utt_len, utt, do_pad=True))
        return np.array(utts)

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss
    
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        if self.config.use_metadata_for_aux_encoder:
            ctx_outs = self.np2var(self.extract_AE_ctx(data_feed), LONG) # contains bs and db
            aux_utt_summary, _, aux_enc_outs = self.aux_encoder(ctx_outs.unsqueeze(1))
        else:
            short_target_utts = self.np2var(data_feed['outputs'], LONG)
            aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.unsqueeze(1)
        aux_enc_last = aux_utt_summary.unsqueeze(1)
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # aux_enc_last = th.cat([bs_label, db_label, aux_utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = aux_log_qy
        else: 
            logits_py, log_py = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)


        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result
    
    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        # pdb.set_trace()
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed), LONG) # contains bs and db
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        enc_last = utt_summary.unsqueeze(1)
        # create decoder initial states
        logits_py, log_qy = self.c2z(enc_last)
        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)
        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        
        if self.config.use_metadata_for_decoding:
            metadata = self.np2var(self.extract_metadata(data_feed), LONG) 
            metadata_summary, _, metadata_enc_outs = self.metadata_encoder(metadata.unsqueeze(1))
            if "metadata_to_decoder" in self.config:
                if self.config.metadata_to_decoder == "add":
                    dec_init_state = dec_init_state + metadata_summary.view(1, batch_size, -1)
                elif self.config.metadata_to_decoder == "avg":
                    dec_init_state = th.mean(th.stack((dec_init_state, metadata_summary.view(1, batch_size, -1))), dim=0)
                else:
                    dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)
            else:
                dec_init_state = th.cat((dec_init_state, metadata_summary.view(1, batch_size, -1)), dim=2)


        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y

class SysE2ECat(BaseModel):
    def __init__(self, corpus, config):
        super(SysE2ECat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        enc_last = utt_summary.squeeze(1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            result['pi_entropy'] = self.entropy_loss(log_qy, unit_average=True)
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        enc_last = utt_summary.squeeze(1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y

class SysE2EActZCat(BaseModel):
    def __init__(self, corpus, config): 
        super(SysE2EActZCat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior # minimize kl to uninformed prior instead of dist conditioned by context
        self.contextual_posterior = config.contextual_posterior # does not use context cause AE task
        if "use_act_label" in config:
            self.use_act_label = config.use_act_label
        else:
            self.use_act_label = False

        if "use_aux_c2z" in config:
            self.use_aux_c2z = config.use_aux_c2z
        else:
            self.use_aux_c2z = False

        self.embedding = None
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)


        if self.use_act_label:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.act_size, config.y_size, config.k_size, is_lstm=False)
        else:
            self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                          config.y_size, config.k_size, is_lstm=False)
            if self.use_aux_c2z:
                self.aux_c2z = nn_lib.Hidden2Discrete(self.aux_encoder.output_size, config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        
        if not self.simple_posterior: #q(z|x,c)
            if self.contextual_posterior:
                # x, c, BS, and DB
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss
    
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        short_target_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = utt_summary.squeeze(1)
        aux_enc_last = aux_utt_summary.squeeze(1)
        # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # aux_enc_last = th.cat([bs_label, db_label, aux_utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            if self.use_aux_c2z:
                aux_logits_qy, aux_log_qy = self.aux_c2z(aux_utt_summary.squeeze(1))
            else:
                aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = aux_log_qy
        else: 
            logits_py, log_py = self.c2z(enc_last)
            aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result
    
    def forward_rl(self, data_feed, max_words, temp=0.1, enc="utt"):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        batch_size = len(ctx_lens)
        if enc == "utt":
            short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)

            utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # create decoder initial states
            enc_last = utt_summary.squeeze(1)
            # create decoder initial states
            if self.simple_posterior:
                logits_py, log_qy = self.c2z(enc_last)
            else:
                logits_py, log_qy = self.c2z(enc_last)

        elif enc == "aux":
            short_target_utts = self.np2var(data_feed['outputs'], LONG)
            # short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['outputs'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)

            aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))
            if self.simple_posterior:
                if self.use_aux_c2z:
                    aux_logits_qy, aux_log_qy = self.aux_c2z(aux_utt_summary.squeeze(1))
                else:
                    aux_enc_last = aux_utt_summary.squeeze(1)
                    aux_logits_qy, aux_log_qy = self.c2z(aux_enc_last)
            logits_py = aux_logits_qy
            log_qy = aux_log_qy

        
        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y

class SysPerfectBD2Gauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Gauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        if "contextual posterior" in config: 
            self.contextual_posterior = config.contextual_posterior
        else:
            self.contextual_posterior = True # default value is true, i.e. q(z|x,c)

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            # self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               # config.y_size, is_lstm=False)
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)

        if "state_for_decoding" not in self.config:
            self.state_for_decoding = False
        else:
            self.state_for_decoding = self.config.state_for_decoding

        if self.state_for_decoding:
            dec_hidden_size = config.dec_cell_size + self.utt_encoder.output_size + self.db_size + self.bs_size
        else:
            dec_hidden_size = config.dec_cell_size



        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=dec_hidden_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.state_for_decoding:
            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z

    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        
        if data_feed:
            ctx_lens = data_feed['context_lens']  # (batch_size, )
            short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
 
        # pack attention context
        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        dec_init_state = self.z_embedding(sample_y.unsqueeze(0))
        if (dec_init_state != dec_init_state).any():
            pdb.set_trace()
        attn_context = None

        # decode
        if self.state_for_decoding:
            utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # create decoder initial states
            enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
            dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)

        return logprobs, outs

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class SysAEGauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysAEGauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        # self.act_size = corpus.act_size
        self.y_size = config.y_size
        self.simple_posterior = True
        self.contextual_posterior = False

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        # if "use_metadata" in self.config and self.config.use_metadata:
        if "ae_zero_padding" in self.config and self.config.ae_zero_padding:
            # self.use_metadata = self.config.use_metadata
            self.ae_zero_padding = self.config.ae_zero_padding
            c2z_input_size = self.utt_encoder.output_size + self.db_size + self.bs_size
        else:
            # self.use_metadata = False
            self.ae_zero_padding = False
            c2z_input_size = self.utt_encoder.output_size

        self.c2z = nn_lib.Hidden2Gaussian(c2z_input_size,
                                          config.y_size, is_lstm=False)

        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
       
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            # self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               # config.y_size, is_lstm=False)
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)


        if "kl_annealing" in self.config and config.kl_annealing=="cyclical":
            if "n_iter" not in self.config:
                config['n_iter'] = config.ckpt_step  * config.max_epoch
            self.beta = frange_cycle_linear(config.n_iter, start=self.config.beta_start, stop=self.config.beta_end, n_cycle=10)    
        else:
            self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0

    def valid_loss(self, loss, batch_cnt=None):
        if isinstance(self.beta, float):
            beta = self.beta
        else:
            if batch_cnt == None:
                beta = self.beta[-1]
            else:
                beta = self.beta[int(batch_cnt)]


        if self.simple_posterior or "kl_annealing" in self.config:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        # print(short_ctx_utts[0])
        # print(out_utts[0])


        # create decoder initial states
        # if self.use_metadata:
        if self.ae_zero_padding:
            enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), utt_summary.squeeze(1)], dim=1)
        else:
            enc_last = utt_summary.squeeze(1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        # else:
            # p_mu, p_logvar = self.c2z(enc_last)
            # # encode response and use posterior to find q(z|x, c)
            # x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            # if self.contextual_posterior:
                # q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            # else:
                # q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            # # use prior at inference time, otherwise use posterior
            # if mode == GEN or use_py:
                # sample_z = self.gauss_connector(p_mu, p_logvar)
            # else:
                # sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            # print(labels[0])
            # print("========")
            # pdb.set_trace()
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

class SysMTGauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysMTGauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior
        if "shared_train" in config:
            self.shared_train = config.shared_train
        else:
            self.shared_train = False

        if "use_aux_kl" in config:
            self.use_aux_kl = config.use_aux_kl
        else:
            self.use_aux_kl = False


        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        # if self.shared_train:
            # self.aux_c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size,
                                          # config.y_size, is_lstm=False)

        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            # self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               # config.y_size, is_lstm=False)
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        if "priornet_config_path" in config and self.config.priornet_config_path is not None:
            self._init_priornet(corpus)
        else:
            self.priornet = None

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.entropy_loss = GaussianEntropy()

        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

        # self.aux_pi_beta = self.config.aux_pi_beta if hasattr(self.config, 'aux_pi_beta') else 1.0
        # if hasattr(self.config, 'aux_pi_beta'):
        if "aux_pi_beta" in self.config:
            self.aux_pi_beta = self.config.aux_pi_beta
        else:
            self.aux_pi_beta = 1.0


    def _init_priornet(self, corpus):
        priornet_config = Pack(json.load(open(self.config.priornet_config_path)))

        if "actz" in self.config.priornet_config_path:
            self.priornet = SysActZGauss(corpus, priornet_config)
        else:
            self.priornet = SysMTGauss(corpus, priornet_config)

        priornet_model_dict = th.load(self.config.priornet_model_path, map_location=lambda storage, location: storage)
        self.priornet.load_state_dict(priornet_model_dict)

        for p in self.priornet.parameters():
            p.requires_grad=False


    def valid_loss(self, loss, batch_cnt=None):
        if self.shared_train:
            if "selective_fine_tune" in self.config and self.config.selective_fine_tune:
                total_loss = loss.nll + self.config.beta * loss.aux_pi_kl
            else:
                total_loss = loss.nll + loss.ae_nll + self.aux_pi_beta * loss.aux_pi_kl + self.config.beta * loss.aux_kl 
        else:
            if self.simple_posterior:
                total_loss = loss.nll
                if self.config.use_pr > 0.0:
                    total_loss += self.config.beta * loss.pi_kl
            else:
                total_loss = loss.nll + loss.pi_kl


        return total_loss
    
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        short_target_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))
        
        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()


        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # aux_enc_last = aux_utt_summary.squeeze(1)
        aux_enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), aux_utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            if self.shared_train:
                # aux_q_mu, aux_q_logvar = self.aux_c2z(aux_enc_last)
                aux_q_mu, aux_q_logvar = self.c2z(aux_enc_last)
                aux_sample_z = self.gauss_connector(aux_q_mu, aux_q_logvar)
            if self.priornet is not None:
                _, p_mu, p_logvar = self.priornet.get_z_via_rg(data_feed)
            else:
                p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            aux_q_mu, aux_q_logvar = self.c2z(aux_enc_last)
            
            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        if self.shared_train:
            aux_dec_init_state = self.z_embedding(aux_sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
            if self.shared_train:
                aux_dec_init_state = tuple([aux_dec_init_state, aux_dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            if self.shared_train:
                ae_dec_outputs, ae_hidden_state, ae_ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=aux_dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
                result['ae_nll'] = self.nll(ae_dec_outputs, labels)
                aux_pi_kl = self.gauss_kl(q_mu, q_logvar, aux_q_mu, aux_q_logvar)
                aux_kl = self.gauss_kl(aux_q_mu, aux_q_logvar, p_mu, p_logvar)
                result['aux_pi_kl'] = aux_pi_kl
                result['aux_kl'] = aux_kl
                # result['aux_entropy'] = self.entropy_loss(aux_q_mu, aux_q_logvar)


            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            # result['pi_entropy'] = self.entropy_loss(q_mu, q_logvar)
            result['nll'] = self.nll(dec_outputs, labels)
            return result
    
    def encode_state(self, data_feed):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        
        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        return enc_last

    def encode_action(self, data_feed):
        batch_size = data_feed.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(data_feed.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = aux_utt_summary.squeeze(1)

        return aux_enc_last
            
    def get_z_via_vae(self, responses):
        batch_size = responses.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(responses.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = th.cat([self.np2var(np.zeros([batch_size, self.bs_size]), LONG), self.np2var(np.zeros([batch_size, self.db_size]), LONG), aux_utt_summary.squeeze(1)], dim=1)

        aux_q_mu, aux_q_logvar = self.c2z(aux_enc_last)
        aux_sample_z = self.gauss_connector(aux_q_mu, aux_q_logvar)
        
        return aux_sample_z, aux_q_mu, aux_q_logvar

    def get_z_via_rg(self, data_feed):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        q_mu, q_logvar = self.c2z(enc_last)

        sample_z = self.gauss_connector(q_mu, q_logvar)
        
        return sample_z, q_mu, q_logvar


    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

        return logprobs, outs, joint_logpz, sample_z

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        # sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        sample_z = self.gauss_connector(p_mu, p_logvar)
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        # if self.state_for_decoding:
            # dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z

    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        
        if data_feed:
            ctx_lens = data_feed['context_lens']  # (batch_size, )
            short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
 
        # pack attention context
        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        dec_init_state = self.z_embedding(sample_y.unsqueeze(0))
        if (dec_init_state != dec_init_state).any():
            pdb.set_trace()
        attn_context = None

        # decode
        # if self.state_for_decoding:
            # if not data_feed:
                # raise ValueError
            # utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # # create decoder initial states
            # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
            # dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)

        return logprobs, outs

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def forward_aux(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False, sample_z=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.aux_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            return result

class SysActZGauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysActZGauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)
        
        self.aux_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)


        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        # self.aux_c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size,
                                          # config.y_size, is_lstm=False)

        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            # self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               # config.y_size, is_lstm=False)
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)


        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        if config.avg_type == "weighted" and config.nll_weight=="no_match_penalty":
            req_tokens = []
            for d in REQ_TOKENS.keys():
                req_tokens.extend(REQ_TOKENS[d])
            nll_weight = Variable(th.FloatTensor([10. if token in req_tokens  else 1. for token in self.vocab]))
            print("req tokens assigned with special weights")
            if config.use_gpu:
                nll_weight = nll_weight.cuda()
            self.nll.set_weight(nll_weight)



        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)
    
    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        if "match_z" in self.config and self.config.match_z:
            total_loss += loss.z_mse

        return total_loss
    
    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        short_target_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(short_target_utts.unsqueeze(1))
        
        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        aux_enc_last = th.cat([bs_label, db_label, aux_utt_summary.squeeze(1)], dim=1)
        # aux_enc_last = aux_utt_summary.squeeze(1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            # p_mu, p_logvar = self.aux_c2z(aux_enc_last)
            p_mu, p_logvar = self.c2z(aux_enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            aux_sample_z = self.gauss_connector(p_mu, p_logvar)
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                q_mu, q_logvar = self.xc2z(x_h.squeeze(1))

            aux_q_mu, aux_q_logvar = self.c2z(aux_enc_last)
            
            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            z_mse = F.mse_loss(aux_sample_z, sample_z)
            result['pi_kl'] = pi_kl
            result['z_mse'] = z_mse
            # result['nll'] = self.nll(dec_outputs, labels)
            return result

    def forward_aux(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        # act_label = self.np2var(data_feed['act'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.aux_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([th.zeros_like(bs_label), th.zeros_like(db_label), utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            ret_dict['q_mu'] = q_mu
            ret_dict['q_logvar'] = q_logvar
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result
    
    def get_z_via_vae(self, responses):
        batch_size = responses.shape[0]
        aux_utt_summary, _, aux_enc_outs = self.aux_encoder(responses.unsqueeze(1))
        
        # create decoder initial states
        aux_enc_last = th.cat([self.np2var(np.zeros([batch_size, self.bs_size]), LONG), self.np2var(np.zeros([batch_size, self.db_size]), LONG), aux_utt_summary.squeeze(1)], dim=1)

        aux_q_mu, aux_q_logvar = self.c2z(aux_enc_last)
        aux_sample_z = self.gauss_connector(aux_q_mu, aux_q_logvar)
        
        return aux_sample_z, aux_q_mu, aux_q_logvar

    def get_z_via_rg(self, data_feed):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        q_mu, q_logvar = self.c2z(enc_last)

        sample_z = self.gauss_connector(q_mu, q_logvar)
        
        return sample_z, q_mu, q_logvar


    def decode_z(self, sample_y, batch_size, data_feed=None, max_words=None, temp=0.1, gen_type='greedy'):
        """
        generate response from latent var
        """
        
        if data_feed:
            ctx_lens = data_feed['context_lens']  # (batch_size, )
            short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
            bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
            db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
 
        # pack attention context
        if isinstance(sample_y, np.ndarray):
            sample_y = self.np2var(sample_y, FLOAT)

        dec_init_state = self.z_embedding(sample_y.unsqueeze(0))
        if (dec_init_state != dec_init_state).any():
            pdb.set_trace()
        attn_context = None

        # decode
        # if self.state_for_decoding:
            # if not data_feed:
                # raise ValueError
            # utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
            # # create decoder initial states
            # enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
            # dec_init_state = th.cat([dec_init_state, enc_last.unsqueeze(0)], dim=2)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)

        return logprobs, outs

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

        return logprobs, outs, joint_logpz, sample_z

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            # print("cutting off, ", tokens)
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens
