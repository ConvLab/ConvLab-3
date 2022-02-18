# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:34 2019

@author: truthless
"""

import pdb


class Environment():

    def __init__(self, sys_nlg, usr, sys_nlu, sys_dst, evaluator=None, use_semantic_acts=False):
        self.sys_nlg = sys_nlg
        self.usr = usr
        self.sys_nlu = sys_nlu
        self.sys_dst = sys_dst
        self.evaluator = evaluator
        self.use_semantic_acts = use_semantic_acts
        self.cur_domain = None

    def reset(self):
        self.usr.init_session()
        self.sys_dst.init_session()
        self.cur_domain = None
        if self.evaluator:
            self.evaluator.add_goal(self.usr.policy.get_goal())
        s, r, t = self.step([])
        return self.sys_dst.state

    def step(self, action):
        if not self.use_semantic_acts:
            model_response = self.sys_nlg.generate(
                action) if self.sys_nlg else action
        else:
            model_response = action
        # If system takes booking action add booking info to the 'book-booked' section of the belief state
        if type(action) == list:
            for intent, domain, slot, value in action:
                if domain.lower() not in ['general', 'booking']:
                    self.cur_domain = domain
                dial_act = f'{domain.lower()}-{intent.lower()}-{slot.lower()}'
                if dial_act == 'booking-book-ref' and self.cur_domain.lower() in ['hotel', 'restaurant', 'train']:
                    if self.cur_domain:
                        self.sys_dst.state['belief_state'][self.cur_domain.lower()]['book']['booked'] = [{slot.lower():value}]
                elif dial_act == 'train-offerbooked-ref' or dial_act == 'train-inform-ref':
                    self.sys_dst.state['belief_state']['train']['book']['booked'] = [{slot.lower():value}]
                elif dial_act == 'taxi-inform-car':
                    self.sys_dst.state['belief_state']['taxi']['book']['booked'] = [{slot.lower():value}]
        observation = self.usr.response(model_response)

        if self.evaluator:
            self.evaluator.add_sys_da(self.usr.get_in_da(), self.sys_dst.state['belief_state'])
            self.evaluator.add_usr_da(self.usr.get_out_da())

        dialog_act = self.sys_nlu.predict(
            observation) if self.sys_nlu else observation
        self.sys_dst.state['user_action'] = dialog_act
        state = self.sys_dst.update(dialog_act)
        dialog_act = self.sys_dst.state['user_action']

        if type(dialog_act) == list:
            for intent, domain, slot, value in dialog_act:
                if domain.lower() not in ['booking', 'general']:
                    self.cur_domain = domain

        state['history'].append(["sys", model_response])
        state['history'].append(["usr", observation])

        terminated = self.usr.is_terminated()

        if self.evaluator:
            reward = self.evaluator.get_reward(terminated)
        else:
            reward = self.usr.get_reward()

        return state, reward, terminated
