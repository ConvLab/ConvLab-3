# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:34 2019

@author: truthless
"""

import pdb
from copy import deepcopy


class Environment():

    def __init__(self, sys_nlg, usr, sys_nlu, sys_dst, evaluator=None, use_semantic_acts=False):
        self.sys_nlg = sys_nlg
        self.usr = usr
        self.sys_nlu = sys_nlu
        self.sys_dst = sys_dst
        self.evaluator = evaluator
        self.use_semantic_acts = use_semantic_acts

    def reset(self, goal=None):
        self.usr.init_session(goal=goal)
        self.sys_dst.init_session()
        if self.evaluator:
            self.evaluator.add_goal(self.usr.policy.get_goal())
        s, r, t = self.step([])
        return self.sys_dst.state

    def step(self, action):
        # save last system action
        self.sys_dst.state['system_action'] = action
        if not self.use_semantic_acts:
            model_response = self.sys_nlg.generate(
                action) if self.sys_nlg else action
        else:
            model_response = action
        # If system takes booking action add booking info to the 'book-booked' section of the belief state
        if type(action) == list:
            for intent, domain, slot, value in action:
                if intent == "book":
                    self.sys_dst.state['booked'][domain] = [{slot: value}]
        observation = self.usr.response(model_response)

        if self.evaluator:
            self.evaluator.add_sys_da(self.usr.get_in_da(), self.sys_dst.state['belief_state'])
            self.evaluator.add_usr_da(self.usr.get_out_da())

        dialog_act = self.sys_nlu.predict(
            observation) if self.sys_nlu else observation
        self.sys_dst.state['user_action'] = dialog_act
        self.sys_dst.state['history'].append(["sys", model_response])
        self.sys_dst.state['history'].append(["user", observation])

        state = self.sys_dst.update(dialog_act)
        self.sys_dst.state['history'].append(["sys", model_response])
        self.sys_dst.state['history'].append(["usr", observation])

        state = deepcopy(state)

        terminated = self.usr.is_terminated()

        if self.evaluator:
            reward = self.evaluator.get_reward(terminated)
        else:
            reward = self.usr.get_reward()

        return state, reward, terminated
