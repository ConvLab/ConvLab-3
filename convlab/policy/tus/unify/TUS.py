import json
import os
import random
from copy import deepcopy

import torch
from convlab.policy.policy import Policy
from convlab.policy.tus.multiwoz.transformer import TransformerActionPrediction
from convlab.policy.tus.unify.Goal import Goal
from convlab.policy.tus.unify.usermanager import BinaryFeature
from convlab.policy.tus.unify.util import create_goal, split_slot_name
from convlab.util import (load_dataset,
                          relative_import_module_from_unified_datasets)
from convlab.util.custom_util import model_downloader
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.policy.tus.unify.Goal import old_goal2list
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal as ABUS_Goal


reverse_da, normalize_domain_slot_value = relative_import_module_from_unified_datasets(
    'multiwoz21', 'preprocess.py', ['reverse_da', 'normalize_domain_slot_value'])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
Inform = "inform"
Request = "request"
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK, ""]


# TODO not ready for unify dataformat now
class UserActionPolicy(Policy):
    def __init__(self, config, pretrain=True, dataset="multiwoz21"):
        Policy.__init__(self)
        self.dataset = dataset
        if isinstance(config, str):
            self.config = json.load(open(config))
        else:
            self.config = config

        feat_type = self.config.get("feat_type", "binary")
        # print("feat_type", feat_type)
        self.feat_handler = BinaryFeature(self.config)

        self.config["num_token"] = config["num_token"]
        self.user = TransformerActionPrediction(self.config).to(device=DEVICE)
        if pretrain:
            model_path = os.path.join(
                self.config["model_dir"], "model-non-zero")  # self.config["model_name"])
            print(f"loading model from {model_path}...")
            self.load(model_path)
        self.user.eval()
        self.use_domain_mask = self.config.get("domain_mask", False)
        self.max_turn = 40
        self.mentioned_domain = []
        self.reward = {"success": 40,
                       "fail": -20}
        self.sys_acts = []
        self.goal_gen = GoalGenerator()
        self.raw_goal = None

    def _no_offer(self, system_in):
        for intent, domain, slot, value in system_in:
            if intent.lower() == "nooffer":
                self.terminated = True
                return True
            else:
                return False

    def predict(self, sys_dialog_act, mode="max"):
        # update goal
        self.predict_action_list = self.goal.action_list(sys_dialog_act)
        cur_state = self.goal.update(action=sys_dialog_act, char="system")
        self.sys_acts.append(sys_dialog_act)

        # need better way to handle this
        if self._no_offer(sys_dialog_act):
            return [["bye", "general", "none", "none"]]

        # update constraint
        self.time_step += 2

        feature, mask = self.feat_handler.get_feature(
            all_slot=self.predict_action_list,
            user_goal=self.goal,
            cur_state=cur_state,
            pre_state=self.sys_history_state,
            sys_action=sys_dialog_act,
            usr_action=self.pre_usr_act)
        feature = torch.tensor([feature], dtype=torch.float).to(DEVICE)
        mask = torch.tensor([mask], dtype=torch.bool).to(DEVICE)

        self.sys_history_state = cur_state

        usr_output = self.user.forward(feature, mask)
        usr_action = self.transform_usr_act(
            usr_output, self.predict_action_list, mode)
        domains = [act[1] for act in usr_action]
        none_slot_acts = self._add_none_slot_act(domains)
        usr_action = none_slot_acts + usr_action

        self.pre_usr_act = deepcopy(usr_action)

        if len(usr_action) < 1:
            print("EMPTY ACTION")

        # convert user action to unify data format
        norm_usr_action = []
        for intent, domain, slot, value in usr_action:
            intent = intent
            # domain, slot, value = normalize_domain_slot_value(
            #     domain, slot, value)
            norm_usr_action.append([intent, domain, slot, value])

        cur_state = self.goal.update(action=norm_usr_action, char="user")

        return norm_usr_action

        # return usr_action

    def init_session(self, goal=None):
        self.mentioned_domain = []
        self.time_step = 0
        self.topic = 'NONE'
        remove_domain = "police"  # remove police domain in inference

        if type(goal) == ABUS_Goal:
            self.raw_goal = goal.domain_goals
            goal_list = old_goal2list(goal.domain_goals)
            goal = Goal(goal_list)
        elif type(goal) == Goal:
            self.raw_goal = goal.domain_goals
        else:
            goal = ABUS_Goal(self.goal_gen)
            self.raw_goal = goal.domain_goals
            goal_list = old_goal2list(goal.domain_goals)
            goal = Goal(goal_list)

        self.read_goal(goal)
        self.feat_handler.initFeatureHandeler(self.goal)

        # print(self.goal)
        if self.config.get("reorder", False):
            self.predict_action_list = self.goal.action_list()
        else:
            self.predict_action_list = self.action_list
        self.sys_history_state = None  # to save sys history
        self.terminated = False

        self.pre_usr_act = None
        self.sys_acts = []

    def read_goal(self, data_goal):
        if type(data_goal) == Goal:
            self.goal = data_goal
        else:
            self.goal = Goal(goal=data_goal)

    # def new_goal(self, remove_domain="police", domain_len=None):
    #     keep_generate_goal = True
    #     while keep_generate_goal:
    #         self.goal = Goal(goal_generator=self.goal_gen)
    #         if (domain_len and len(self.goal.domains) != domain_len) or \
    #                 (remove_domain and remove_domain in self.goal.domains):
    #             keep_generate_goal = True
    #         else:
    #             keep_generate_goal = False

    def load(self, model_path=None):
        self.user.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def load_state_dict(self, model=None):
        self.user.load_state_dict(model)

    def _get_goal(self):
        # internal usage
        return self.goal.domain_goals

    def get_goal(self):
        # for outside usage, e.g. evaluator
        return self.raw_goal

    def get_reward(self):
        if self.goal.task_complete():
            # reward = 2 * self.max_turn
            reward = self.reward["success"]
            # reward = 1

        elif self.time_step >= self.max_turn:
            # reward = -1 * self.max_turn
            reward = self.reward["fail"]
            # reward = -1

        else:
            # reward = -1.0
            reward = 0
        return reward

    def _add_none_slot_act(self, domains):
        actions = []
        for domain in domains:
            domain = domain.lower()
            if domain not in self.mentioned_domain and domain != 'general':
                actions.append([Inform, domain, "none", "none"])
                self.mentioned_domain.append(domain)
        return actions

    def _finish_conversation(self):

        if self.goal.task_complete():
            return True, [['thank', 'general', 'none', 'none']]

        if self.time_step > self.max_turn:
            return True, [["bye", "general", "none", "none"]]

        if len(self.sys_acts) >= 3:
            if self.sys_acts[-1] == self.sys_acts[-2] and self.sys_acts[-2] == self.sys_acts[-3]:
                return True, [["bye", "general", "none", "none"]]

        return False, [[]]

    def transform_usr_act(self, usr_output, action_list, mode="max"):
        is_finish, usr_action = self._finish_conversation()
        if is_finish:
            self.terminated = True
            # if "bye" == usr_action[0][0]:
            #     print("fail")
            #     pprint(self.goal.domain_goals)
            #     pprint(self.goal.status)
            return usr_action

        usr_action = self._get_acts(
            usr_output, action_list, mode)

        # if usr_action is empty, sample at least one
        while not usr_action:
            usr_action = self._get_acts(
                usr_output, action_list, mode="pick-one")

        if self.use_domain_mask:
            domain_mask = self._get_prediction_domain(torch.round(
                torch.sigmoid(usr_output[0, 0, :])).tolist())
            usr_action = self._mask_user_action(usr_action, domain_mask)

        return usr_action

    def _get_acts(self, usr_output, action_list, mode="max"):
        score = {}
        for index, slot_name in enumerate(action_list):
            weights = self.user.softmax(usr_output[0, index + 1, :])
            if mode == "max":
                o = torch.argmax(usr_output[0, index + 1, :]).item()

            elif mode == "sample" or mode == "pick-one":
                o = random.choices(
                    range(self.config["out_dim"]),
                    weights=weights,
                    k=1)
                o = o[0]
            else:
                print("(BUG) unknown mode")
            v = weights[o]
            score[slot_name] = {"output": o, "weight": v}

        usr_action = self._append_actions(action_list, score)

        if mode == "sample" and len(usr_action) > 3:
            slot_names = []
            outputs = []
            scores = []
            for index, slot_name in enumerate(action_list):
                weights = self.user.softmax(usr_output[0, index + 1, :])
                o = torch.argmax(usr_output[0, index + 1, 1:]).item() + 1
                slot_names.append(slot_name)
                outputs.append(o)
                scores.append(weights[o].item())
            slot_name = random.choices(
                slot_names,
                weights=scores,
                k=3)
            slot_name = slot_name[0]
            score[slot_name]["output"] = outputs[slot_names.index(slot_name)]
            score[slot_name]["weight"] = scores[slot_names.index(slot_name)]
            # print(score)
            usr_action = self._append_actions(action_list, score)

        if mode == "pick-one" and not usr_action:
            # print("pick-one")
            slot_names = []
            outputs = []
            scores = []
            for index, slot_name in enumerate(action_list):
                weights = self.user.softmax(usr_output[0, index + 1, :])
                o = torch.argmax(usr_output[0, index + 1, 1:]).item() + 1
                slot_names.append(slot_name)
                outputs.append(o)
                scores.append(weights[o].item())
            slot_name = random.choices(
                slot_names,
                weights=scores,
                k=1)
            slot_name = slot_name[0]
            score[slot_name]["output"] = outputs[slot_names.index(slot_name)]
            score[slot_name]["weight"] = scores[slot_names.index(slot_name)]
            # print(score)
            usr_action = self._append_actions(action_list, score)

        return usr_action

    def _append_actions(self, action_list, score):
        usr_action = []
        for index, slot_name in enumerate(action_list):
            domain, slot = split_slot_name(slot_name)
            is_action, act = self._add_user_action(
                output=score[slot_name]["output"],
                domain=domain,
                slot=slot)
            if is_action:
                usr_action += act
        return usr_action

    def _mask_user_action(self, usr_action, mask):
        mask_action = []
        for intent, domain, slot, value in usr_action:
            if domain.lower() in mask:
                mask_action += [[intent, domain, slot, value]]
        return mask_action

    def _get_prediction_domain(self, domain_output):
        predict_domain = []
        if domain_output[0] > 0:
            predict_domain.append('general')
        for index, value in enumerate(domain_output[1:]):
            if value > 0 and index < len(self.goal.domains):
                predict_domain.append(self.goal.domains[index])
        return predict_domain

    def _add_user_action(self, output, domain, slot):
        goal = self._get_goal()
        is_action = False
        act = [[]]
        value = None

        # get intent
        if output == 1:
            intent = Request
        else:
            intent = Inform

        # "?"
        if output == 1:  # "?"
            value = DEF_VAL_UNK

        # "dontcare"
        elif output == 2:
            value = DEF_VAL_DNC

        # system
        elif output == 3 and domain in self.sys_history_state:
            value = self.sys_history_state[domain].get(
                slot, "")

        elif output == 4 and domain in goal:  # usr
            for slot_type in ["info"]:
                if slot_type in goal[domain] and slot in goal[domain][slot_type]:
                    value = goal[domain][slot_type][slot]

        # elif output == 5 and domain.lower() in goal:
        #     if domain.lower() not in self.all_values["all_value"]:
        #         value = None
        #     elif slot.lower() not in self.all_values["all_value"][domain.lower()]:
        #         value = None
        #     else:
        #         value = random.choice(
        #             list(self.all_values["all_value"][domain.lower()][slot.lower()].keys()))

        if value:
            is_action, act = self._form_action(
                intent, domain, slot, value)

        return is_action, act

    def _get_action_slot(self, domain, slot):
        return slot

    def _form_action(self, intent, domain, slot, value):
        action_slot = self._get_action_slot(domain, slot)
        if action_slot:
            return True, [[intent, domain, action_slot, value]]
        return False, [[]]

    def is_terminated(self):
        # Is there any action to say?
        return self.terminated

    def _slot_type(self, domain, slot):
        slot_type = ""
        if slot in self.sys_history_state[domain]["book"]:
            slot_type = "book"
        elif slot in self.sys_history_state[domain]["semi"]:
            slot_type = "semi"

        return slot_type


class UserPolicy(Policy):
    def __init__(self, config, dial_ids_order=0):
        if isinstance(config, str):
            self.config = json.load(open(config))
        else:
            self.config = config
        self.config["model_dir"] = f'{self.config["model_dir"]}_{dial_ids_order}'
        print("model_dir", self.config['model_dir'])
        if not os.path.exists(self.config["model_dir"]):
            # os.mkdir(self.config["model_dir"])
            model_downloader(os.path.dirname(self.config["model_dir"]),
                             "https://zenodo.org/record/7369429/files/multiwoz_0.zip")
        self.slot2dbattr = {
            'open hours': 'openhours',
            'price range': 'pricerange',
            'arrive by': 'arriveBy',
            'leave at': 'leaveAt',
            'train id': 'trainID'
        }
        self.dbattr2slot = {}
        for k, v in self.slot2dbattr.items():
            self.dbattr2slot[v] = k

        self.policy = UserActionPolicy(self.config)

    def predict(self, state):
        raw_act = self.policy.predict(state)
        act = []
        for intent, domain, slot, value in raw_act:
            if slot in self.dbattr2slot:
                slot = self.dbattr2slot[slot]
            act.append([intent, domain, slot, value])
        return act

    def init_session(self, goal=None):
        self.policy.init_session(goal)

    def is_terminated(self):
        return self.policy.is_terminated()

    def get_reward(self):
        return self.policy.get_reward()

    def get_goal(self):
        if hasattr(self.policy, 'get_goal'):
            return self.policy.get_goal()
        return None
