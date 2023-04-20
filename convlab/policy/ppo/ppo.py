# -*- coding: utf-8 -*-
import torch
from torch import optim
import random
import numpy as np
import logging
import os
import json
from convlab.policy.vector.vector_binary import VectorBinary
from convlab.policy.policy import Policy
from convlab.policy.rlmodule import MultiDiscretePolicy, Value
from convlab.util.custom_util import model_downloader, set_seed
import sys
import urllib.request


root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO(Policy):

    def __init__(self, is_train=False, seed=0, vectorizer=None, load_path="", **kwargs):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs' ,'ppo_config.json'), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), cfg['save_dir'])
        self.cfg = cfg
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.optim_batchsz = cfg['batchsz']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.tau = cfg['tau']
        self.is_train = is_train
        self.info_dict = {}
        self.vector = vectorizer

        logging.info('PPO seed ' + str(seed))
        set_seed(seed)
        dir_name = os.path.dirname(os.path.abspath(__file__))

        if self.vector is None:
            logging.info("No vectorizer was set, using default..")
            self.vector = VectorBinary(dataset_name=kwargs['dataset_name'],
                         use_masking=kwargs.get('use_masking', True),
                         manually_add_entity_names=kwargs.get('manually_add_entity_names', True),
                         seed=seed)

        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'],
                                          self.vector.da_dim, seed).to(device=DEVICE)
        logging.info(f"ACTION DIM OF PPO: {self.vector.da_dim}")
        logging.info(f"STATE DIM OF PPO: {self.vector.state_dim}")

        try:
            if load_path == "from_pretrained":
                urllib.request.urlretrieve(
                    f"https://huggingface.co/ConvLab/mle-policy-{self.vector.dataset_name}/resolve/main/supervised.pol.mdl",
                    f"{dir_name}/{self.vector.dataset_name}_mle.pol.mdl")
                load_path = f"{dir_name}/{self.vector.dataset_name}_mle"
            self.load_policy(load_path)
        except Exception as e:
            print(f"Could not load the policy, Exception: {e}")

        self.value = Value(self.vector.state_dim,
                           cfg['hv_dim']).to(device=DEVICE)
        if is_train:
            self.policy_optim = optim.RMSprop(
                self.policy.parameters(), lr=cfg['policy_lr'])
            self.value_optim = optim.Adam(
                self.value.parameters(), lr=cfg['value_lr'])

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """

        s, action_mask = self.vector.state_vectorize(state)
        s_vec = torch.Tensor(s)
        mask_vec = torch.Tensor(action_mask)
        a = self.policy.select_action(
            s_vec.to(device=DEVICE), False, action_mask=mask_vec.to(device=DEVICE)).cpu()

        a_counter = 0
        while a.sum() == 0:
            a_counter += 1
            a = self.policy.select_action(
                s_vec.to(device=DEVICE), True, action_mask=mask_vec.to(device=DEVICE)).cpu()
            if a_counter == 5:
                break
        # print('True :')
        # print(a)
        action = self.vector.action_devectorize(a.detach().numpy())
        self.info_dict["action_used"] = action
        # for key in state.keys():
        #     print("Key : {} , Value : {}".format(key,state[key]))
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def est_adv(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)

        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]

        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target

    def update(self, epoch, batchsz, s, a, r, mask, action_mask):
        # get estimated V(s) and PI_old(s, a)
        # actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed
        # v: [b, 1] => [b]
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a, action_mask).detach()

        # estimate advantage and v_target according to GAE and Bellman Equation
        A_sa, v_target = self.est_adv(r, v, mask)

        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)
            # shuffle the variable for mutliple optimize
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf, action_mask_shuf = \
                v_target[perm], A_sa[perm], s[perm], a[perm], log_pi_old_sa[perm], action_mask[perm]

            # 2. get mini-batch for optimizing
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf, action_mask_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                torch.chunk(A_sa_shuf, optim_chunk_num), \
                torch.chunk(s_shuf, optim_chunk_num), \
                torch.chunk(a_shuf, optim_chunk_num), \
                torch.chunk(log_pi_old_sa_shuf,
                            optim_chunk_num), \
                torch.chunk(action_mask_shuf, optim_chunk_num)
            # 3. iterate all mini-batch to optimize
            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b, action_mask_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                                    log_pi_old_sa_shuf, action_mask_shuf):

                # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
                # 1. update value network
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()

                # backprop
                loss.backward()
                self.value_optim.step()

                # 2. update policy network by clipping
                self.policy_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.policy.get_log_prob(s_b, a_b, action_mask_b)

                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                # because the joint action prob is the multiplication of the prob of each da
                # it may become extremely small
                # and the ratio may be inf in this case, which causes the gradient to be nan
                # clamp in case of the inf ratio, which causes the gradient to be nan
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(
                    ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # although the ratio is clamped, the grad may still contain nan due to 0 * inf
                # set the inf in the gradient to 0
                for p in self.policy.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
                # self.lock.acquire() # retain lock to update weights
                self.policy_optim.step()

                # self.lock.release() # release lock

            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num
            # print("valueloss " + str(value_loss))
            # print("policyloss" + str(policy_loss))
            # if (epoch + 1) % self.save_per_epoch == 0:
            # self.save(self.save_dir, epoch)

    def save(self, directory, addition="", best_complete_rate=False, best_success_rate=False):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.value.state_dict(), directory +
                   f'/{addition}_ppo.val.mdl')
        torch.save(self.policy.state_dict(), directory +
                   f'/{addition}_ppo.pol.mdl')

        logging.info(f"Saved policy and critic.")

    # Function to load model object weights from binary files
    def load(self, filename):
        value_mdl_candidates = [
            filename + '.val.mdl',
            filename + '_ppo.val.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.val.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.val.mdl')
        ]
        for value_mdl in value_mdl_candidates:
            if os.path.exists(value_mdl):
                self.value.load_state_dict(torch.load(value_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(value_mdl))
                break

        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_ppo.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

    def load_policy(self, filename=""):
        policy_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_ppo.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_ppo.pol.mdl')
        ]
        for policy_mdl in policy_mdl_candidates:
            if os.path.exists(policy_mdl):
                print(f"Loaded policy checkpoint from file: {policy_mdl}")
                self.policy.load_state_dict(torch.load(policy_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
                break

    # Load model from model_path(URL)
    def load_from_pretrained(self, model_path=""):

        model_path = model_path if model_path != "" else \
            "https://zenodo.org/record/5783185/files/supervised_MLP.zip"

        # Get download directory
        download_path = os.path.dirname(os.path.abspath(__file__))
        download_path = os.path.join(download_path, 'pretrained_models')
        # Downloadable model path format http://.../ppo_model_name.zip
        filename = model_path.split('/')[-1].replace('.zip', '')
        filename = os.path.join(download_path, filename, 'save', 'supervised')
        # Check if model file exists
        exists = [os.path.exists(filename + '.pol.mdl'),
                  os.path.exists(filename + '_ppo.pol.mdl'),
                  os.path.exists(os.path.join(filename, 'best_ppo.pol.mdl'))]
        exists = True in exists

        if not exists:
            if not os.path.exists(download_path):
                os.mkdir(download_path)
            model_downloader(download_path, model_path)

        # Once downloaded use the load function to load from binaries
        self.load(filename)

    @staticmethod
    def load_vectoriser(name):
        if name == 'base':
            from convlab.policy.vector.vector_binary import MultiWozVector
            return MultiWozVector()

    @classmethod
    def from_pretrained(cls,
                        model_file="",
                        is_train=False,
                        dataset='Multiwoz',
                        vectoriser='base'):
        vector = cls.load_vectoriser(vectoriser)
        model = cls(is_train=is_train, dataset=dataset, vectorizer=vector)
        model.load_from_pretrained(model_file)
        return model
