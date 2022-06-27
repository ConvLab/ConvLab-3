# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:34 2019

@author: truthless
"""

import pdb
from convlab.dialog_agent.env import Environment


class UsrEnvironment(Environment):

    def __init__(self, sys, usr, evaluator=None, use_semantic_acts=False):
        self.sys = sys
        self.usr = usr
        self.evaluator = evaluator

    def reset(self):
        self.usr.init_session()
        self.sys.init_session()
        if self.evaluator:
            self.evaluator.add_goal(self.usr.policy.get_goal())
        sys_response = self.sys.response([])
        usr_response = self.usr.response(sys_response)
        s, r, t = self.step(usr_response)
        # return self.sys_dst.state
        print("-" * 20)

        return self.usr.dst.state

    def step(self, action):

        # if not self.use_semantic_acts:
        #     model_response = self.sys_nlg.generate(
        #         action) if self.sys_nlg else action
        # else:
        #     model_response = action

        # only semantic level
        usr_response = action
        sys_response = self.sys.response(usr_response)
        print(f"(env_usr) usr: {usr_response}")
        print(f"(env_usr) sys: {sys_response}")

        if self.evaluator:
            if not self.usr.get_in_da():
                # print("not sure why")
                usr_in_da = sys_response
                usr_out_da = action
            else:
                usr_in_da = self.usr.get_in_da()
                usr_out_da = self.usr.get_out_da()

            # print(f"usr_in_da {usr_in_da}, usr_out_da {usr_out_da}")

            self.evaluator.add_sys_da(usr_in_da)
            self.evaluator.add_usr_da(usr_out_da)

        # dialog_act = self.sys_nlu.predict(
        #     observation) if self.sys_nlu else observation
        # TODO pipeline agent should update the dst itself <- make sure why
        state = self.usr.dst.update(sys_response)
        self.usr.dst.state['user_action'] = usr_response
        self.usr.dst.state['system_action'] = sys_response
        self.usr.dst.state['history'].append(["usr", usr_response])
        self.usr.dst.state['history'].append(["sys", sys_response])

        terminated = self.usr.is_terminated()

        if terminated:
            # TODO uncomment this line
            # if self.evaluator:
            #     if self.evaluator.task_success():
            #         reward = 80/40
            #     elif self.evaluator.cur_domain and self.evaluator.domain_success(self.evaluator.cur_domain):
            #         reward = 0
            #     else:
            #         reward = -40/40
            # else:
            reward = self.usr.get_reward()
        else:

            # reward = -1 + self.usr.policy.get_turn_reward()
            # reward = reward / 40
            reward = self.usr.policy.get_turn_reward()

        return state, reward, terminated
