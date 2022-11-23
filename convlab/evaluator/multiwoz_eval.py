# -*- coding: utf-8 -*-

import logging
import re
import numpy as np
import pdb

from copy import deepcopy
from data.unified_datasets.multiwoz21.preprocess import reverse_da, reverse_da_slot_name_map
from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from convlab.evaluator.evaluator import Evaluator
from data.unified_datasets.multiwoz21.preprocess import reverse_da_slot_name_map
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import unified_format, act_dict_to_flat_tuple
from convlab.util.multiwoz.dbquery import Database
from convlab.util import relative_import_module_from_unified_datasets

# import reflect table
REF_SYS_DA_M = {}
for dom, ref_slots in REF_SYS_DA.items():
    dom = dom.lower()
    REF_SYS_DA_M[dom] = {}
    for slot_a, slot_b in ref_slots.items():
        if slot_a == 'Ref':
            slot_b = 'ref'
        REF_SYS_DA_M[dom][slot_a.lower()] = slot_b
    REF_SYS_DA_M[dom]['none'] = 'none'
REF_SYS_DA_M['taxi']['phone'] = 'phone'
REF_SYS_DA_M['taxi']['car'] = 'car type'

reverse_da = relative_import_module_from_unified_datasets('multiwoz21', 'preprocess.py', 'reverse_da')


requestable = \
    {'attraction': ['post', 'phone', 'addr', 'fee', 'area', 'type'],
     'restaurant': ['addr', 'phone', 'post', 'ref', 'price', 'area', 'food'],
     'train': ['ticket', 'time', 'ref', 'id', 'arrive', 'leave'],
     'hotel': ['addr', 'post', 'phone', 'ref', 'price', 'internet', 'parking', 'area', 'type', 'stars'],
     'taxi': ['car', 'phone'],
     'hospital': ['post', 'phone', 'addr'],
     'police': ['addr', 'post', 'phone']}

belief_domains = requestable.keys()

mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'price': 'pricerange', 'ref': 'ref'},
           'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name',
                     'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type', 'ref': 'ref'},
           'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'type': 'type'},
           'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination',
                     'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price', 'ref': 'ref'},
           'taxi': {'car': 'car type', 'phone': 'phone'},
           'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
           'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}


time_re = re.compile(r'^(([01]\d|2[0-4]):([0-5]\d)|24:00)$')
NUL_VALUE = ["", "dont care", 'not mentioned',
             "don't care", "dontcare", "do n't care"]
REF_SYS_DA_M = {}
for dom, ref_slots in REF_SYS_DA.items():
    dom = dom.lower()
    REF_SYS_DA_M[dom] = {}
    for slot_a, slot_b in ref_slots.items():
        if slot_a == 'Ref':
            slot_b = 'ref'
        REF_SYS_DA_M[dom][slot_a.lower()] = slot_b
    REF_SYS_DA_M[dom]['none'] = 'none'
REF_SYS_DA_M['taxi']['phone'] = 'phone'
REF_SYS_DA_M['taxi']['car'] = 'car type'
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked

NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]

# Not sure values in inform
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]


class MultiWozEvaluator(Evaluator):
    def __init__(self, check_book_constraints=True, check_domain_success=False):
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = {}
        self.cur_domain = ''
        self.booked = {}
        self.database = Database()
        self.dbs = self.database.dbs
        self.check_book_constraints = check_book_constraints
        self.check_domain_success = check_domain_success
        self.complete = 0
        self.success = 0
        self.success_strict = 0
        self.successful_domains = []
        logging.info(
            f"We check booking constraints: {self.check_book_constraints}")

    def _init_dict(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = {'info': {}, 'book': {}, 'reqt': []}
        return dic

    def _init_dict_booked(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = None
        return dic

    def _expand(self, _goal):
        goal = deepcopy(_goal)
        for domain in belief_domains:
            if domain not in goal:
                goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
                continue
            if 'info' not in goal[domain]:
                goal[domain]['info'] = {}
            if 'book' not in goal[domain]:
                goal[domain]['book'] = {}
            if 'reqt' not in goal[domain]:
                goal[domain]['reqt'] = []
        return goal

    def add_goal(self, goal):
        """init goal and array

        args:
            goal:
                dict[domain] dict['info'/'book'/'reqt'] dict/dict/list[slot]
        """
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = deepcopy(goal)
        self.cur_domain = ''
        self.booked = self._init_dict_booked()
        self.booked_states = self._init_dict_booked()
        self.successful_domains = []

    @staticmethod
    def _convert_action(act):
        act = unified_format(act)
        act = reverse_da(act)
        act = act_dict_to_flat_tuple(act)
        return act

    def add_sys_da(self, da_turn, belief_state=None):
        """add sys_da into array

        args:
            da_turn:
                list[intent, domain, slot, value]
        """

        new_acts = list()
        for intent, domain, slot, value in da_turn:
            if intent.lower() == 'book':
                ref = [_value for _intent, _domain, _slot, _value in da_turn if _domain == domain and _intent.lower() == 'inform' and _slot.lower() == 'ref']
                ref = ref[0] if ref else ''
                value = ref
            new_acts.append([intent, domain, slot, value])
        da_turn = new_acts

        da_turn = self._convert_action(da_turn)

        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.sys_da_array.append(da + '-' + value)

            # new booking actions make life easier
            if intent.lower() == "book":
                # taxi has no DB queries
                if domain.lower() == "taxi":
                    if not self.booked['taxi']:
                        self.booked['taxi'] = 'booked'
                else:
                    if not self.booked[domain] and re.match(r'^\d{8}$', value) and \
                            len(self.dbs[domain]) > int(value):
                        self.booked[domain] = self.dbs[domain][int(
                            value)].copy()
                        self.booked[domain]['Ref'] = value
                        if belief_state is not None:
                            self.booked_states[domain] = deepcopy(
                                belief_state[domain])
                        else:
                            self.booked_states[domain] = None
        self.goal = self.update_goal(self.goal, da_turn)

    def add_usr_da(self, da_turn):
        """add usr_da into array

        args:
            da_turn:
                list[intent, domain, slot, value]
        """
        da_turn = self._convert_action(da_turn)
        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.usr_da_array.append(da + '-' + value)

    def _book_rate_goal(self, goal, booked_entity, domains=None):
        """
        judge if the selected entity meets the informable constraint
        """
        if domains is None:
            domains = belief_domains
        score = []
        for domain in domains:
            if 'book' in goal[domain] and goal[domain]['book']:
                tot = len(goal[domain]['info'].keys())
                if tot == 0:
                    continue
                entity = booked_entity[domain]
                if entity is None:
                    score.append(0)
                    continue
                if domain == 'taxi':
                    score.append(1)
                    continue
                match = 0
                for k, v in goal[domain]['info'].items():
                    if k in ['destination', 'departure']:
                        tot -= 1
                    elif k == 'leaveAt':
                        try:
                            v_constraint = int(
                                v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(
                                ':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arriveBy':
                        try:
                            v_constraint = int(
                                v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(
                                entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[k].strip():
                            match += 1
                if tot != 0:
                    score.append(match / tot)
        return score

    def _book_goal_constraints(self, goal, booked_states, domains=None):
        """
        judge if the selected entity meets the booking constraint
        """
        if domains is None:
            domains = belief_domains
        score = []
        for domain in domains:
            if domain == "taxi":
                # taxi has no booking constraints
                continue
            if 'book' in goal[domain] and goal[domain]['book']:
                tot = len(goal[domain]['book'].keys())
                if tot == 0:
                    continue
                state = booked_states.get(domain, None)
                if state is None:
                    # nothing has been booked but should have been
                    score.append(0)
                    continue
                tracks_booking = False
                for slot in state:
                    if "book" in slot:
                        tracks_booking = True
                        break
                if not tracks_booking:
                    # state does not track any booking constraints -> trivially satisfied
                    score.append(1)
                    continue

                match = 0
                for slot, value in goal[domain]['book'].items():
                    try:
                        value_predicted = state.get(f"book {slot}", "")
                        if value == value_predicted:
                            match += 1
                    except Exception as e:
                        print("Tracker probably does not track that slot.", e)
                        # if tracker does not track it, it trivially matches since policy has no chance otherwise
                        match += 1

                if tot != 0:
                    score.append(match / tot)
        return score

    def _inform_F1_goal(self, goal, sys_history, domains=None):
        """
        judge if all the requested information is answered
        """
        if domains is None:
            domains = belief_domains
        inform_slot = {}
        for domain in domains:
            inform_slot[domain] = set()
        TP, FP, FN = 0, 0, 0

        inform_not_reqt = set()
        reqt_not_inform = set()
        bad_inform = set()
        for da in sys_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and \
                    domain in domains and slot in mapping[domain] and value.strip() not in NUL_VALUE:
                key = mapping[domain][slot]
                if self._check_value(domain, key, value):
                    # print('add key', key)
                    inform_slot[domain].add(key)
                else:
                    bad_inform.add((intent, domain, key))
                    FP += 1
        for domain in domains:
            for k in goal[domain]['reqt']:
                if k in inform_slot[domain]:
                    # print('k: ', k)
                    TP += 1
                else:
                    # print('FN + 1')
                    reqt_not_inform.add(('request', domain, k))
                    FN += 1
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal[domain]['reqt'] \
                        and k not in goal[domain]['info'] \
                        and k in requestable[domain]:
                    # print('FP + 1 @2', k)
                    inform_not_reqt.add(('inform', domain, k,))
                    FP += 1
        return TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt

    def _check_value(self, domain, key, value):
        if key == "area":
            return value.lower() in ["centre", "east", "south", "west", "north"]
        elif key == "arriveBy" or key == "leaveAt":
            return time_re.match(value)
        elif key == "day":
            return value.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday",
                                     "saturday", "sunday"]
        elif key == "duration":
            return 'minute' in value
        elif key == "internet" or key == "parking":
            return value in ["yes", "no", "none"]
        elif key == "phone":
            return re.match(r'^\d{11}$', value) or domain == "restaurant"
        elif key == "price":
            return 'pound' in value
        elif key == "pricerange":
            return value in ["cheap", "expensive", "moderate", "free"] or domain == "attraction"
        elif key == "postcode":
            return re.match(r'^cb\d{1,3}[a-z]{2,3}$', value) or value == 'pe296fl'
        elif key == "stars":
            return re.match(r'^\d$', value)
        elif key == "trainID":
            return re.match(r'^tr\d{4}$', value.lower())
        else:
            return True

    def book_rate(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
        score = self._book_rate_goal(goal, self.booked)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def book_rate_constrains(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
        score = self._book_goal_constraints(goal, self.booked_states)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def check_booking_done(self, ref2goal=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = self.goal[domain]['book']

        # check for every domain where booking is required whether a booking has been made
        for domain in goal:
            if goal[domain]['book']:
                if not self.booked[domain]:
                    return False

        return True

    def inform_F1(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt = self._inform_F1_goal(
            goal, self.sys_da_array)
        if aggregate:
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
        else:
            return [TP, FP, FN]

    def task_success(self, ref2goal=True):
        """
        judge if all the domains are successfully completed
        """
        booking_done = self.check_booking_done(ref2goal)
        book_sess = self.book_rate(ref2goal)
        book_constraint_sess = self.book_rate_constrains(ref2goal)
        inform_sess = self.inform_F1(ref2goal)
        goal_sess = self.final_goal_analyze()

        if ((book_sess == 1 and inform_sess[1] == 1)
            or (book_sess == 1 and inform_sess[1] is None)
            or (book_sess is None and inform_sess[1] == 1)) \
                and goal_sess == 1:
            self.complete = 1
            self.success = 1
            self.success_strict = 1 if (
                book_constraint_sess == 1 or book_constraint_sess is None) else 0
            return self.success if not self.check_book_constraints else self.success_strict
        else:
            self.complete = 1 if booking_done and (
                inform_sess[1] == 1 or inform_sess[1] is None) else 0
            self.success = 0
            self.success_strict = 0
            return 0

    def domain_reqt_inform_analyze(self, domain, ref2goal=True):
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        return inform

    def domain_success(self, domain, ref2goal=True):
        """
        judge if the domain (subtask) is successfully completed
        """
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        book_constraints = self._book_goal_constraints(
            goal, self.booked_states, [domain])
        book_constraints = np.mean(
            book_constraints) if book_constraints else None

        book_rate = self._book_rate_goal(goal, self.booked, [domain])
        book_rate = np.mean(book_rate) if book_rate else None
        match, mismatch = self._final_goal_analyze_domain(domain)
        goal_sess = 1 if (match == 0 and mismatch ==
                          0) else match / (match + mismatch)

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        try:
            inform_rec = inform[0] / (inform[0] + inform[2])
        except ZeroDivisionError:
            inform_rec = None

        if ((book_rate == 1 and inform_rec == 1) or (book_rate == 1 and inform_rec is None) or
                (book_rate is None and inform_rec == 1)) and goal_sess == 1:
            domain_success = 1
            domain_strict_success = 1 if (
                book_constraints == 1 or book_constraints is None) else 0
            return domain_success if not self.check_book_constraints else domain_strict_success
        else:
            return 0

    def _final_goal_analyze_domain(self, domain):

        match = mismatch = 0
        if domain in self.goal:
            dom_goal_dict = self.goal[domain]
        else:
            return match, mismatch
        constraints = []
        if 'reqt' in dom_goal_dict:
            reqt_constraints = list(dom_goal_dict['reqt'].items())
            constraints += reqt_constraints
        else:
            reqt_constraints = []
        if 'info' in dom_goal_dict:
            info_constraints = list(dom_goal_dict['info'].items())
            constraints += info_constraints
        else:
            info_constraints = []
        query_result = self.database.query(
            domain, info_constraints + reqt_constraints)
        if not query_result:
            mismatch += 1

        booked = self.booked[domain]
        if not self.goal[domain].get('book'):
            match += 1
        elif isinstance(booked, dict):
            ref = booked['Ref']
            if any(found['Ref'] == ref for found in query_result):
                match += 1
            else:
                mismatch += 1
        else:
            match += 1
        return match, mismatch

    def _final_goal_analyze(self):
        """whether the final goal satisfies constraints"""
        match = mismatch = 0
        for domain, dom_goal_dict in self.goal.items():
            constraints = []
            if 'reqt' in dom_goal_dict:
                reqt_constraints = list(dom_goal_dict['reqt'].items())
                constraints += reqt_constraints
            else:
                reqt_constraints = []
            if 'info' in dom_goal_dict:
                info_constraints = list(dom_goal_dict['info'].items())
                constraints += info_constraints
            else:
                info_constraints = []
            query_result = self.database.query(
                domain, info_constraints + reqt_constraints)
            if not query_result:
                mismatch += 1
                continue

            booked = self.booked[domain]
            if not self.goal[domain].get('book'):
                match += 1
            elif isinstance(booked, dict):
                ref = booked['Ref']
                if any(found['Ref'] == ref for found in query_result):
                    match += 1
                else:
                    mismatch += 1
            else:
                match += 1
        return match, mismatch

    def final_goal_analyze(self):
        """percentage of domains, in which the final goal satisfies the database constraints.
        If there is no dialog action, returns 1."""
        match, mismatch = self._final_goal_analyze()
        if match == mismatch == 0:
            return 1
        else:
            return match / (match + mismatch)

    def get_reward(self, terminated=False):

        if terminated:
            # once dialogue ended check task success
            if self.task_success():
                reward = 80
            else:
                reward = -40
        else:
            reward = -1

            if self.check_domain_success and not self.task_success():
                if self.cur_domain and self.domain_success(self.cur_domain) and \
                        self.cur_domain not in self.successful_domains:
                    # if domain is in successful_domains, the domain_success reward has been already given
                    reward += 5
                    self.successful_domains.append(self.cur_domain)

        return reward

    def evaluate_dialog(self, goal, user_acts, system_acts, system_states):

        self.add_goal(goal.domain_goals)
        for sys_act, sys_state, user_act in zip(system_acts, system_states, user_acts):
            self.add_sys_da(sys_act, sys_state)
            self.add_usr_da(user_act)
        self.task_success()
        return {"complete": self.complete, "success": self.success, "success_strict": self.success_strict}

    def update_goal(self, goal, system_action):
        for intent, domain, slot, val in system_action:
            # need to reverse slot to old representation
            if slot in reverse_da_slot_name_map:
                slot = reverse_da_slot_name_map[slot]
            elif domain in reverse_da_slot_name_map and slot in reverse_da_slot_name_map[domain]:
                slot = reverse_da_slot_name_map[domain][slot]
            else:
                slot = slot.capitalize()
            if intent.lower() in ['inform', 'recommend']:
                if domain.lower() in goal:
                    if 'reqt' in goal[domain.lower()]:
                        if REF_SYS_DA_M.get(domain.lower(), {}).get(slot.lower(), slot.lower()) \
                                in goal[domain.lower()]['reqt']:
                            if val in NOT_SURE_VALS:
                                val = '\"' + val + '\"'
                            goal[domain.lower()]['reqt'][
                                REF_SYS_DA_M.get(domain.lower(), {}).get(slot.lower(), slot.lower())] = val
        return goal
