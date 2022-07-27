'''
SubjectiveFeedbackManager.py - Update policy according to subjective feedback
==========================================================================

@author: Songbo, Chris

'''

from configparser import ConfigParser
import torch
import pickle
import os
import time
import logging

# from convlab2.util.train_util import save_to_bucket


class SubjectiveFeedbackManager(object):

    def __init__(self, configPath, policy, agent_name=""):
        self.sys_dialogue_utterance = []
        self.sys_dialogue_state_vec = []
        self.sys_action_mask_vec = []
        self.sys_dialogue_act_vec = []
        self.sys_dialogue_reward_vec = []
        self.sys_dialogue_mask_vec = []
        self.agent_name = agent_name
        configparser = ConfigParser()
        configparser.read(configPath)
        self.turn_reward = int(configparser.get("SUBJECTIVE", "turnReward"))
        self.subject_reward = int(
            configparser.get("SUBJECTIVE", "subjectReward"))
        self.updatePerSession = int(
            configparser.get("SUBJECTIVE", "updatePerSession"))
        self.memory = Memory()

        # All policy update is done by this instances.
        self.policy = policy
        self.add_counter = 0
        self.add_counter_total = 0

        self.trainingEpoch = int(
            configparser.get("SUBJECTIVE", "trainingEpoch"))

        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     f"policy/AMT/AMT_REAL_{agent_name}_{current_time}")
        os.makedirs(self.save_dir, exist_ok=True)

    def init_list(self):
        self.sys_dialogue_utterance = []
        self.sys_dialogue_state_vec = []
        self.sys_dialogue_act_vec = []
        self.sys_dialogue_reward_vec = []
        self.sys_dialogue_mask_vec = []
        self.sys_action_mask_vec = []
        self.add_counter = 0

    def get_reward_vector(self, state_list, act_list, isGoalAchieved):
        assert len(state_list) == len(act_list)
        reward_vector = []
        for i in range(1, len(state_list)):
            reward_vector.append(self.turn_reward)
        if isGoalAchieved:
            reward_vector.append(2 * self.subject_reward)
        else:
            reward_vector.append(-self.subject_reward)
        return reward_vector

    def get_mask_vector(self, state_list, act_list):
        assert len(state_list) == len(act_list)
        mask_vector = []
        for i in range(1, len(state_list)):
            mask_vector.append(1)
        mask_vector.append(0)
        return mask_vector

    def add_state_action_lists(self, utterance_list, state_list, act_list, isGoalAchieved, reward_list, task_id,
                               system_outputs, prob_history):
        assert len(state_list) == len(act_list) and isGoalAchieved in [True, False]

        self.sys_dialogue_utterance.extend(utterance_list)

        state_list_vec = []
        action_mask_vec = []
        for s in state_list:
            s_vec, mask_ = self.policy.vector.state_vectorize(
                s, output_mask=True)
            mask_ = mask_ + [0, 0]
            state_list_vec.append(s_vec)
            action_mask_vec.append(mask_)

        self.sys_dialogue_state_vec.extend(state_list_vec)
        self.sys_action_mask_vec.extend(action_mask_vec)

        reward_list_new = self.get_reward_vector(
            state_list, act_list, isGoalAchieved)

        try:
            action_list_vec = list(
                map(self.policy.vector.action_vectorize, act_list))
            self.sys_dialogue_act_vec.extend(action_list_vec)
        except:
            # we assume the acts are already action_vectorized
            self.sys_dialogue_act_vec.extend(act_list)

        # TODO: Change the reward here!!
        self.sys_dialogue_reward_vec.extend(reward_list_new)

        self.sys_dialogue_mask_vec.extend(
            self.get_mask_vector(state_list, act_list))
        self.add_counter += 1
        self.add_counter_total += 1

        logging.info(
            f"Added dialog, we now have {self.add_counter} dialogs in total.")

        try:
            if hasattr(self.policy, "last_action"):
                if len(state_list_vec) == 0 or len(act_list) == 0 or len(reward_list) == 0:
                    pass
                else:
                    self.policy.update_memory(
                        utterance_list, state_list_vec, act_list, reward_list_new)
                    self.memory.add_experience(utterance_list, state_list, state_list_vec, act_list, reward_list,
                                               isGoalAchieved, task_id, system_outputs, prob_history)
            else:
                self.policy.update_memory(
                    utterance_list, state_list_vec, action_list_vec, reward_list_new)
                self.memory.add_experience(utterance_list, state_list, state_list_vec, action_list_vec, reward_list,
                                           isGoalAchieved, task_id, system_outputs, prob_history)
        except:
            pass
        print("Session Added to FeedbackManager {}".format(self.add_counter))
        # if(self.add_counter % self.updatePerSession == 0 and self.add_counter > 400):
        if (self.add_counter % self.updatePerSession == 0 and self.add_counter > 0):

            logging.info("Manager updating policy.")
            try:
                self.updatePolicy()
                logging.info("Successfully updated policy.")
            except Exception as e:
                logging.info("Couldnt update policy. Exception: ", e)

        logging.info("Saving AMT memory")
        self.memory.save(self.save_dir)
        try:
            self.save_into_bucket()
        except:
            print("SubjectiveFeedbackManager: Could not save into bucket")

    def updatePolicy(self):
        try:
            train_state_list = torch.Tensor(self.sys_dialogue_state_vec)
            train_act_list = torch.Tensor(self.sys_dialogue_act_vec)
            train_reward_list = torch.Tensor(self.sys_dialogue_reward_vec)
            train_mask_list = torch.Tensor(self.sys_dialogue_mask_vec)
            train_action_mask_list = torch.Tensor(self.sys_action_mask_vec)
            batchsz = train_state_list.size()[0]
        except:
            train_state_list = (self.sys_dialogue_state_vec)
            train_act_list = (self.sys_dialogue_act_vec)
            train_reward_list = (self.sys_dialogue_reward_vec)
            train_mask_list = (self.sys_dialogue_mask_vec)
            train_action_mask_list = (self.sys_action_mask_vec)
            batchsz = 32

        for i in range(self.trainingEpoch):
            # print(train_state_list)
            # print(train_action_mask_list)
            # print(train_act_list)
            # print(train_reward_list)
            self.policy.update(i, batchsz, train_state_list, train_act_list, train_reward_list, train_mask_list,
                               train_action_mask_list)
        if self.policy.is_train:
            self.policy.save(self.save_dir)

            if self.add_counter_total % 200 == 0:
                self.policy.save(
                    self.save_dir, addition=f"_{self.add_counter}")

        # Empty the current batch. This is needed for on-policy algorithms like PPO.
        self.init_list()

    def getUpdatedPolicy(self):
        return self.policy

    def save_into_bucket(self):
        print(f"Saving into bucket from {self.save_dir}")
        bucket_save_name = self.save_dir.split('/')[-1]
        for file in os.listdir(self.save_dir):
            save_to_bucket('geishauser', f'AMT_Experiments/{bucket_save_name}/{file}',
                           os.path.join(self.save_dir, file))


class Memory:

    def __init__(self):
        self.utterances = []
        self.raw_states = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.feedback = []
        self.task_id = []
        self.sys_outputs = []
        self.action_probs = []

    def add_experience(self, utterances, raw_states, states, actions, rewards, feedback, task_id, system_outputs,
                       prob_history):
        self.utterances.append(utterances)
        self.raw_states.append(raw_states)
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.feedback.append(feedback)
        self.task_id.append(task_id)
        self.sys_outputs.append(system_outputs)
        self.action_probs.append(prob_history)

    def save(self, directory):
        with open(directory + '/' + 'AMT_memory.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
