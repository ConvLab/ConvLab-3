# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import os
import json, random
import torch
import pickle

import logging
from queue import PriorityQueue

from convlab.util.custom_util import set_seed


class Memory:

    def __init__(self, seed=0):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)

        self.batch_size = cfg.get('batchsz', 32)
        self.max_size = cfg.get('memory_size', 2000)
        self.reservoir_sampling = cfg.get("use_reservoir_sampling", False)
        logging.info(f"We use reservoir sampling: {self.reservoir_sampling}")
        self.second_r = False
        self.reward_weight = 1.0
        self.priority_queue = PriorityQueue()

        self.size = 0  # total experiences stored
        self.number_episodes = 0

        self.data_keys = ['states', 'actions', 'rewards', 'small_actions', 'mu', 'action_masks', 'critic_value',
                          'description_idx_list', 'value_list', 'current_domain_mask', 'non_current_domain_mask']
        self.reset()
        set_seed(seed)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def reset(self):
        for k in self.data_keys:
            setattr(self, k, [[]])

    def update_episode(self, state_list, action_list, reward_list, small_act_list, mu_list, action_mask_list,
                       critic_value_list, description_idx_list, value_list, current_domain_mask, non_current_domain_mask):

        if len(self.states) > self.max_size:
            # delete the oldest episode when max-size is reached
            #for k in self.data_keys:
            #    getattr(self, k).pop(0)
            if not self.reservoir_sampling:
                # We sample a random experience for deletion
                remove_index = random.choice(range(len(self.states) - 2))
            else:
                item = self.priority_queue.get()
                remove_index = item[1]

            for k in self.data_keys:
                getattr(self, k).pop(remove_index)

        self.states[-1] = state_list
        self.actions[-1] = action_list
        self.rewards[-1] = [r/40.0 for r in reward_list]
        self.small_actions[-1] = small_act_list
        self.mu[-1] = mu_list
        self.action_masks[-1] = action_mask_list
        self.critic_value[-1] = critic_value_list
        self.description_idx_list[-1] = description_idx_list
        self.value_list[-1] = value_list
        self.current_domain_mask[-1] = current_domain_mask
        self.non_current_domain_mask[-1] = non_current_domain_mask

        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.small_actions.append([])
        self.mu.append([])
        self.action_masks.append([])
        self.critic_value.append([])
        self.description_idx_list.append([])
        self.value_list.append([])
        self.current_domain_mask.append([])
        self.non_current_domain_mask.append([])

        self.number_episodes += 1

        if self.reservoir_sampling:
            self.priority_queue.put((torch.randn(1), len(self.states) - 2))

    def update(self, state, action, reward, next_state, done):

        self.add_experience(state, action, reward, next_state, done)

    def add_experience(self, state, action, reward, next_state, done, mu=None):

        reward = reward / 40.0
        if isinstance(action, dict):
            mu = action.get('mu')
            action_index = action.get('action_index')
            mask = action.get('mask')
        else:
            action_index = action

        if done:
            self.states[-1].append(state)
            self.actions[-1].append(action_index)
            self.rewards[-1].append(reward)
            self.next_states[-1].append(next_state)
            #self.dones[-1].append(done)
            self.mu[-1].append(mu)
            self.masks[-1].append(mask)

            self.states.append([])
            self.actions.append([])
            self.rewards.append([])
            self.next_states.append([])
            #self.dones.append([])
            self.mu.append([])
            self.masks.append([])

            if len(self.states) > self.max_size:
                #self.number_episodes = self.max_size
                #delete the oldest episode when max-size is reached
                for k in self.data_keys:
                    getattr(self, k).pop(0)
            else:
                self.number_episodes += 1

        else:
            self.states[-1].append(state)
            self.actions[-1].append(action_index)
            self.rewards[-1].append(reward)
            self.next_states[-1].append(next_state)
            #self.dones[-1].append(done)
            self.mu[-1].append(mu)
            self.masks[-1].append(mask)

        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1

    def sample(self, online_offline_ratio=0.0):
        '''
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        '''
        number_episodes = len(self.states) - 1
        num_online = 0

        #Sample batch-size many episodes
        if number_episodes <= self.batch_size:
            batch_ids = list(range(number_episodes))
        elif online_offline_ratio != 0:
            num_online = int(online_offline_ratio * self.batch_size)
            batch_ids_online = list(range(number_episodes - num_online, number_episodes - 1))
            batch_ids_offline = np.random.randint(number_episodes - 1 - num_online, size=self.batch_size - num_online).tolist()
            batch_ids = batch_ids_online + batch_ids_offline
        else:
            batch_ids = np.random.randint(number_episodes - 1, size=self.batch_size).tolist()

        batch = {}
        for k in self.data_keys:
            batch[k] = [getattr(self, k)[index] for index in batch_ids]

        return batch, num_online

    def save(self, path):

        # PriorityQueue is not serializable, so only save the list behind it
        self.priority_queue = self.priority_queue.queue
        with open(path + f'/vtrace.memory', "wb") as f:
            pickle.dump(self, f)

    def build_priority_queue(self, queue_list):

        self.priority_queue = PriorityQueue()
        for element in queue_list:
            self.priority_queue.put(element)
