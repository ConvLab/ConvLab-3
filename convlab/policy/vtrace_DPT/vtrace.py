import numpy as np
import logging
import json
import os
import sys
import torch
import torch.nn as nn

from torch import optim
from convlab.policy.vtrace_DPT.transformer_model.EncoderDecoder import EncoderDecoder
from convlab.policy.vtrace_DPT.transformer_model.EncoderCritic import EncoderCritic
from ... import Policy
from ...util.custom_util import set_seed

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VTRACE(nn.Module, Policy):

    def __init__(self, is_train=True, seed=0, vectorizer=None, load_path=""):

        super(VTRACE, self).__init__()

        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(dir_name, 'config.json')

        with open(self.config_path, 'r') as f:
            cfg = json.load(f)

        self.cfg = cfg
        self.save_dir = os.path.join(dir_name, cfg['save_dir'])
        self.save_per_epoch = cfg['save_per_epoch']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.is_train = is_train
        self.entropy_weight = cfg.get('entropy_weight', 0.0)
        self.behaviour_cloning_weight = cfg.get('behaviour_cloning_weight', 0.0)
        self.online_offline_ratio = cfg.get('online_offline_ratio', 0.0)
        self.hidden_size = cfg['hidden_size']
        self.policy_freq = cfg['policy_freq']
        self.seed = seed
        self.total_it = 0
        self.rho_bar = cfg.get('rho_bar', 10)
        self.c = cfg['c']
        self.info_dict = {}
        self.use_regularization = False
        self.supervised_weight = cfg.get('supervised_weight', 0.0)

        logging.info(f"Entropy weight: {self.entropy_weight}")
        logging.info(f"Online-Offline-ratio: {self.online_offline_ratio}")
        logging.info(f"Behaviour cloning weight: {self.behaviour_cloning_weight}")
        logging.info(f"Supervised weight: {self.supervised_weight}")

        set_seed(seed)

        self.last_action = None

        self.vector = vectorizer
        self.policy = EncoderDecoder(**self.cfg, action_dict=self.vector.act2vec).to(device=DEVICE)
        self.value_helper = EncoderDecoder(**self.cfg, action_dict=self.vector.act2vec).to(device=DEVICE)

        try:
            self.load_policy(load_path)
        except Exception as e:
            print(f"Could not load the policy, Exception: {e}")

        if self.cfg['independent']:
            self.value = EncoderCritic(self.value_helper.node_embedder, self.value_helper.encoder, **self.cfg).to(
                device=DEVICE)
        else:
            self.value = EncoderCritic(self.policy.node_embedder, self.policy.encoder, **self.cfg).to(device=DEVICE)

        try:
            self.load_value(load_path)
        except Exception as e:
            print(f"Could not load the critic, Exception: {e}")

        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': cfg['policy_lr'], 'betas': (0.0, 0.999)},
            {'params': self.value.parameters(), 'lr': cfg['value_lr']}
        ])

        try:
            self.load_optimizer_dicts(load_path)
        except Exception as e:
            print(f"Could not load optimiser dicts, Exception: {e}")

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """

        if not self.is_train:
            for param in self.policy.parameters():
                param.requires_grad = False
            for param in self.value.parameters():
                param.requires_grad = False

        s, action_mask = self.vector.state_vectorize(state)

        kg_states = [self.vector.kg_info]
        a = self.policy.select_action(kg_states, mask=action_mask, eval=not self.is_train).detach().cpu()
        self.info_dict = self.policy.info_dict

        descr_list = self.info_dict["description_idx_list"]
        value_list = self.info_dict["value_list"]
        current_domain_mask = self.info_dict["current_domain_mask"].unsqueeze(0)
        non_current_domain_mask = self.info_dict["non_current_domain_mask"].unsqueeze(0)

        a_prob, _ = self.policy.get_prob(a.unsqueeze(0), self.info_dict['action_mask'].unsqueeze(0),
                                         len(self.info_dict['small_act']), [self.info_dict['small_act']],
                                         current_domain_mask, non_current_domain_mask, [descr_list], [value_list])

        self.info_dict['big_act'] = a
        self.info_dict['a_prob'] = a_prob.prod()
        self.info_dict['critic_value'] = self.value([descr_list], [value_list]).squeeze()

        action = self.vector.action_devectorize(a.detach().numpy())

        return action

    def update(self, memory):
        p_loss, v_loss = self.get_loss(memory)
        loss = v_loss
        if p_loss is not None:
            loss += p_loss

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 40)
        for p in self.policy.parameters():
            if p.grad is not None:
                p.grad[p.grad != p.grad] = 0.0
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)

        self.optimizer.step()

    def get_loss(self, memory):

        self.is_train = True

        if self.is_train:
            self.total_it += 1

            for param in self.policy.parameters():
                param.requires_grad = True
            for param in self.value.parameters():
                param.requires_grad = True

            batch, num_online = self.get_batch(memory)

            action_masks, actions, critic_v, current_domain_mask, description_batch, max_length, mu, \
            non_current_domain_mask, rewards, small_actions, unflattened_states, value_batch \
                = self.prepare_batch(batch)

            with torch.no_grad():
                values = self.value(description_batch, value_batch).squeeze(-1)

                pi_prob, _ = self.policy.get_prob(actions, action_masks, max_length, small_actions,
                                                  current_domain_mask, non_current_domain_mask,
                                                  description_batch, value_batch)
                pi_prob = pi_prob.prod(dim=-1)

                rho = torch.min(torch.Tensor([self.rho_bar]).to(DEVICE), pi_prob / mu)
                cs = torch.min(torch.Tensor([self.c]).to(DEVICE), pi_prob / mu)

                vtrace_target, advantages = self.compute_vtrace_advantage(unflattened_states, rewards, rho, cs, values)

            # Compute critic loss
            current_v = self.value(description_batch, value_batch).to(DEVICE)
            critic_loss = torch.square(vtrace_target.unsqueeze(-1).to(DEVICE) - current_v).mean()

            if self.use_regularization:
                # do behaviour cloning on the buffer data
                num_online = sum([len(reward_list) for reward_list in batch['rewards'][:num_online]])

                behaviour_loss_critic = torch.square(
                    critic_v[num_online:].unsqueeze(-1).to(DEVICE) - current_v[num_online:]).mean()
                critic_loss += self.behaviour_cloning_weight * behaviour_loss_critic

            actor_loss = None

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                actor_loss, entropy = self.policy.get_log_prob(actions, action_masks, max_length, small_actions,
                                                               current_domain_mask, non_current_domain_mask,
                                                               description_batch, value_batch)
                actor_loss = -1 * actor_loss
                actor_loss = actor_loss * (advantages.to(DEVICE) * rho)
                actor_loss = actor_loss.mean() - entropy * self.entropy_weight

                if self.use_regularization:
                    log_prob, entropy = self.policy.get_log_prob(actions[num_online:], action_masks[num_online:],
                                                                 max_length, small_actions[num_online:],
                                                                 current_domain_mask[num_online:],
                                                                 non_current_domain_mask[num_online:],
                                                                 description_batch[num_online:],
                                                                 value_batch[num_online:])
                    actor_loss = actor_loss - log_prob.mean() * self.behaviour_cloning_weight

            return actor_loss, critic_loss

        else:
            return np.nan

    def get_batch(self, memory):

        if self.use_regularization or self.online_offline_ratio == 1.0:
            batch, num_online = memory.sample(self.online_offline_ratio)
        else:
            batch, num_online = memory.sample(0.0)
        return batch, num_online

    def prepare_batch(self, batch):
        unflattened_states = batch['states']
        states = [kg for kg_list in unflattened_states for kg in kg_list]
        description_batch = batch['description_idx_list']
        description_batch = [descr_ for descr_episode in description_batch for descr_ in descr_episode]
        value_batch = batch['value_list']
        value_batch = [value_ for value_episode in value_batch for value_ in value_episode]

        current_domain_mask = batch['current_domain_mask']
        current_domain_mask = torch.stack([curr_mask for curr_mask_episode in current_domain_mask
                                           for curr_mask in curr_mask_episode]).to(DEVICE)
        non_current_domain_mask = batch['non_current_domain_mask']
        non_current_domain_mask = torch.stack([non_curr_mask for non_curr_mask_episode in non_current_domain_mask
                                               for non_curr_mask in non_curr_mask_episode]).to(DEVICE)
        actions = batch['actions']
        actions = torch.stack([act for act_list in actions for act in act_list], dim=0).to(DEVICE)
        small_actions = batch['small_actions']
        small_actions = [act for act_list in small_actions for act in act_list]
        rewards = batch['rewards']
        rewards = torch.stack([r for r_episode in rewards for r in r_episode]).to(DEVICE)
        # rewards = torch.from_numpy(np.concatenate(np.array(rewards), axis=0)).to(DEVICE)
        mu = batch['mu']
        mu = torch.stack([mu_ for mu_list in mu for mu_ in mu_list], dim=0).to(DEVICE)
        critic_v = batch['critic_value']
        critic_v = torch.stack([v for v_list in critic_v for v in v_list]).to(DEVICE)
        max_length = max(len(act) for act in small_actions)
        action_masks = batch['action_masks']
        action_mask_list = [mask for mask_list in action_masks for mask in mask_list]
        action_masks = torch.stack([torch.cat([
            action_mask.to(DEVICE),
            torch.zeros(max_length - len(action_mask), len(self.policy.action_embedder.small_action_dict)).to(
                DEVICE)],
            dim=0) for action_mask in action_mask_list]).to(DEVICE)
        return action_masks, actions, critic_v, current_domain_mask, description_batch, max_length, mu, \
               non_current_domain_mask, rewards, small_actions, unflattened_states, value_batch

    def compute_vtrace_advantage(self, states, rewards, rho, cs, values):

        vtraces, advantages, offset = [], [], 0
        #len(states) is number of episodes sampled, so we iterate over episodes
        for j in range(0, len(states)):
            vtrace_list, advantage_list, new_vtrace, v_next = [], [], 0, 0
            for i in range(len(states[j]) - 1, -1, -1):
                v_now = values[offset + i]
                delta = rewards[offset + i] + self.gamma * v_next - v_now
                delta = rho[offset + i] * delta
                advantage = rewards[offset + i] + self.gamma * new_vtrace - v_now
                new_vtrace = v_now + delta + self.gamma * cs[offset + i] * (new_vtrace - v_next)
                v_next = v_now
                vtrace_list.append(new_vtrace)
                advantage_list.append(advantage)
            vtrace_list = list(reversed(vtrace_list))
            advantange_list = list(reversed(advantage_list))
            vtraces.append(vtrace_list)
            advantages.append(advantange_list)
            offset += len(states[j])

        vtraces_flat = torch.Tensor([v for v_episode in vtraces for v in v_episode])
        advantages_flat = torch.Tensor([a for a_episode in advantages for a in a_episode])
        return vtraces_flat, advantages_flat

    def save(self, directory, addition=""):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.value.state_dict(), directory + f'/{addition}_vtrace.val.mdl')
        torch.save(self.policy.state_dict(), directory + f'/{addition}_vtrace.pol.mdl')
        torch.save(self.optimizer.state_dict(), directory + f'/{addition}_vtrace.optimizer')

        logging.info(f"Saved policy, critic and optimizer.")

    def load(self, filename):

        value_mdl_candidates = [
            filename + '.val.mdl',
            filename + '_vtrace.val.mdl',
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), filename + '.val.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(
                __file__)), filename + '_vtrace.val.mdl')
        ]
        for value_mdl in value_mdl_candidates:
            if os.path.exists(value_mdl):
                self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
                print('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
                break

        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_vtrace.pol.mdl',
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(
                __file__)), filename + '_vtrace.pol.mdl')
        ]

        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                self.value_helper.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                print('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

    def load_policy(self, filename):

        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_vtrace.pol.mdl',
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(
                __file__)), filename + '_vtrace.pol.mdl')
        ]

        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                self.value_helper.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

    def load_value(self, filename):

        value_mdl_candidates = [
            filename + '.val.mdl',
            filename + '_vtrace.val.mdl',
            os.path.join(os.path.dirname(
                os.path.abspath(__file__)), filename + '.val.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(
                __file__)), filename + '_vtrace.val.mdl')
        ]
        for value_mdl in value_mdl_candidates:
            if os.path.exists(value_mdl):
                self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
                break

    def load_optimizer_dicts(self, filename):
        self.optimizer.load_state_dict(torch.load(filename + f".optimizer", map_location=DEVICE))
        logging.info('<<dialog policy>> loaded optimisers from file: {}'.format(filename))

    def from_pretrained(self):
        raise NotImplementedError

    def init_session(self):
        pass
