import json
import os
from random import shuffle

from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.policy.tus.multiwoz.Da2Goal import SysDa2Goal, UsrDa2Goal
from convlab.policy.tus.multiwoz.util import parse_user_goal
from convlab.task.multiwoz.goal_generator import GoalGenerator

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK, ""]

ref_slot_data2stand = {
    'train': {
        'duration': 'time', 'price': 'ticket', 'trainid': 'id'
    }
}


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal_generator=None, goal=None):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        self.max_domain_len = 5
        self.max_slot_len = 20
        self.local_id = {}
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/all_value.json')
        self.all_values = json.load(open(path))
        if goal_generator is not None and goal is None:
            self.domain_goals = goal_generator.get_user_goal()
            self.domains = list(self.domain_goals['domain_ordering'])
            del self.domain_goals['domain_ordering']
        elif goal_generator is None and goal is not None:
            self.domains = []
            self.domain_goals = {}
            for domain in goal.domains:
                if domain in SysDa2Goal and goal.domain_goals[domain]:  # TODO check order
                    self.domains.append(domain)
                    self.domain_goals[domain] = goal.domain_goals[domain]
        else:
            print("Warning!!! One of goal_generator or goal should not be None!!!")

        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {
                    slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}

            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK
        self.init_local_id()
        self.init_info_record()
        self.actions = None
        self.evaluator = MultiWozEvaluator()
        self.evaluator.add_goal(self.domain_goals)

    def init_info_record(self):
        self.info = {}
        for domain in self.domains:
            if 'info' in self.domain_goals[domain].keys():
                self.info[domain] = {}
                for slot in self.domain_goals[domain]['info']:
                    self.info[domain][slot] = DEF_VAL_NUL

    def add_sys_da(self, sys_act):
        self.evaluator.add_sys_da(sys_act)

    def add_usr_da(self, usr_act):
        self.evaluator.add_usr_da(usr_act)

    def _update_info_action(self, act):
        for intent, domain, slot, value in act:
            domain = domain.lower()
            value = value.lower()
            slot = slot.lower()
            domain, slot = self._norm_domain_slot(domain, slot, value)
            if domain in self.info and slot in self.info[domain] and value not in NOT_SURE_VALS:
                self.info[domain][slot] = value

    def init_local_id(self):
        # local_id = {
        #     "domain 1": {
        #         "ID": [1, 0, 0],
        #         "SLOT": {
        #             "slot 1": [1, 0, 0],
        #             "slot 2": [0, 1, 0]}}}

        for domain_id, domain in enumerate(self.domains):
            self._init_domain_id(domain)
            self._update_domain_id(domain, domain_id)
            slot_id = 0
            for slot_type in ["info", "book", "reqt"]:
                for slot in self.domain_goals[domain].get(slot_type, {}):
                    self._init_slot_id(domain, slot)
                    self._update_slot_id(domain, slot, slot_id)
                    slot_id += 1

    def insert_local_id(self, new_slot_name):
        domain, slot = new_slot_name.split('-')
        if domain not in self.local_id:
            self._init_domain_id(domain)
            domain_id = len(self.domains) + 1
            self._update_domain_id(domain, domain_id)
            self._init_slot_id(domain, slot)
            # the first slot for a new domain
            self._update_slot_id(domain, slot, 0)

        else:
            slot_id = len(self.local_id[domain]["SLOT"]) + 1
            self._init_slot_id(domain, slot)
            self._update_slot_id(domain, slot, slot_id)

    def get_slot_id(self, slot_name):
        domain, slot = slot_name.split('-')
        if domain in self.local_id and slot in self.local_id[domain]["SLOT"]:
            return self.local_id[domain]["ID"], self.local_id[domain]["SLOT"][slot]
        else:  # a slot not in original user goal
            self.insert_local_id(slot_name)
            domain_id, slot_id = self.get_slot_id(slot_name)
            return domain_id, slot_id

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        return self.evaluator.task_success()

    def next_domain_incomplete(self):
        # request
        for domain in self.domains:
            # reqt
            if 'reqt' in self.domain_goals[domain]:
                requests = self.domain_goals[domain]['reqt']
                unknow_reqts = [
                    key for (key, val) in requests.items() if val in NOT_SURE_VALS]
                if len(unknow_reqts) > 0:
                    return domain, 'reqt', ['name'] if 'name' in unknow_reqts else unknow_reqts

            # book
            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return domain, 'book', \
                        self.domain_goals[domain]['fail_book'] if 'fail_book' in self.domain_goals[domain].keys() else \
                        self.domain_goals[domain]['book']

        return None, None, None

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def action_list(self, user_history=None, sys_act=None, all_values=None):
        goal_slot = parse_user_goal(self)

        if not self.actions:
            if user_history:
                self.actions = self._reorder_based_on_user_history(
                    user_history, goal_slot)

            else:
                # priority_action = [slot for slot in goal_slot]
                self.actions = self._reorder_random(goal_slot)
        priority_action = self.actions

        if sys_act:
            for intent, domain, slot, value in sys_act:
                slot_name = self.act2slot(
                    intent, domain, slot, value, all_values)
                # print("system_mention:", slot_name)
                if slot_name and slot_name not in priority_action:
                    priority_action.insert(0, slot_name)

        return priority_action

    def get_booking_domain(self, slot, value, all_values):
        for domain in self.domains:
            if slot in all_values["all_value"] and value in all_values["all_value"][slot]:
                return domain
        print("NOT FOUND BOOKING DOMAIN")
        return ""

    def act2slot(self, intent, domain, slot, value, all_values):
        domain = domain.lower()
        slot = slot.lower()

        if domain not in UsrDa2Goal:
            # print(f"Not handle domain {domain}")
            return ""

        if domain == "booking":
            slot = SysDa2Goal[domain][slot]
            domain = self.get_booking_domain(slot, value, all_values)
            if domain:
                return f"{domain}-{slot}"

        elif domain in UsrDa2Goal:
            if slot in SysDa2Goal[domain]:
                slot = SysDa2Goal[domain][slot]
            elif slot in UsrDa2Goal[domain]:
                slot = UsrDa2Goal[domain][slot]
            elif slot in SysDa2Goal["booking"]:
                slot = SysDa2Goal["booking"][slot]

            return f"{domain}-{slot}"
        return ""

    def _reorder_random(self, goal_slot):
        new_order = [slot for slot in goal_slot]
        return new_order

    def _reorder_based_on_user_history(self, user_history, goal_slot):
        # user_history = [slot_0, slot_1, ...]
        new_order = []
        for slot in user_history:
            if slot and slot not in new_order:
                new_order.append(slot)

        for slot in goal_slot:
            if slot not in new_order:
                new_order.append(slot)
        return new_order

    def update_user_goal(self, action=None, state=None):
        # update request and booked
        if action:
            self._update_user_goal_from_action(action)
        if state:
            self._update_user_goal_from_state(state)
            self._check_booked(state)  # this should always check

        if action is None and state is None:
            print("Warning!!!! Both action and state are None")

    def _check_booked(self, state):
        for domain in self.domains:
            if "booked" in self.domain_goals[domain]:
                if self._check_book_info(state, domain):
                    self.domain_goals[domain]["booked"] = DEF_VAL_BOOKED
                else:
                    self.domain_goals[domain]["booked"] = DEF_VAL_NOBOOK

    def _check_book_info(self, state, domain):
        if domain not in state:
            return False

        for slot_type in ['info', 'book']:
            for slot in self.domain_goals[domain].get(slot_type, {}):
                user_value = self.domain_goals[domain][slot_type][slot]
                if slot in state[domain]["semi"]:
                    state_value = state[domain]["semi"][slot]

                elif slot in state[domain]["book"]:
                    state_value = state[domain]["book"][slot]
                else:
                    state_value = ""
                # only check mentioned values (?)
                if state_value and state_value != user_value:
                    return False

        return True

    def _update_user_goal_from_action(self, action):
        for intent, domain, slot, value in action:
            # print("update user goal from action")
            # print(intent, domain, slot, value)
            # print("action:", intent)
            domain = domain.lower()
            value = value.lower()
            slot = slot.lower()
            if slot == "ref":  # TODO ref!!!! not bug free!!!!
                for usr_domain in self.domains:
                    if "booked" in self.domain_goals[usr_domain]:
                        self.domain_goals[usr_domain]["booked"] = DEF_VAL_BOOKED
            else:
                domain, slot = self._norm_domain_slot(domain, slot, value)

                if self._check_update_request(domain, slot) and value != "?":
                    self.domain_goals[domain]['reqt'][slot] = value
                    # print(f"update reqt {slot} = {value} from system action")

    def _norm_domain_slot(self, domain, slot, value):
        if domain == "booking":
            # ["book", "booking", "people", 7]
            if slot in SysDa2Goal[domain]:
                slot = SysDa2Goal[domain][slot]
                domain = self._get_booking_domain(slot, value)
            else:
                domain = ""
                for d in SysDa2Goal:
                    if slot in SysDa2Goal[d]:
                        domain = d
                        slot = SysDa2Goal[d][slot]
            if not domain:  # TODO make sure what happened!
                return "", ""
            return domain, slot

        elif domain in self.domains:
            if slot in SysDa2Goal[domain]:
                # ["request", "restaurant", "area", "north"]
                slot = SysDa2Goal[domain][slot]
            elif slot in UsrDa2Goal[domain]:
                slot = UsrDa2Goal[domain][slot]
            elif slot in SysDa2Goal["booking"]:
                # ["inform", "hotel", "stay", 2]
                slot = SysDa2Goal["booking"][slot]

            return domain, slot

        else:
            # domain = general
            return "", ""

    def _update_user_goal_from_state(self, state):
        for domain in state:
            for slot in state[domain]["semi"]:
                if self._check_update_request(domain, slot):
                    self._update_user_goal_from_semi(state, domain, slot)
            for slot in state[domain]["book"]:
                if slot == "booked" and state[domain]["book"]["booked"]:
                    self._update_booked(state, domain)

                elif state[domain]["book"][slot] and self._check_update_request(domain, slot):
                    self._update_book(state, domain, slot)

    def _update_slot(self, domain, slot, value):
        self.domain_goals[domain]['reqt'][slot] = value

    def _update_user_goal_from_semi(self, state, domain, slot):
        if self._check_value(state[domain]["semi"][slot]):
            self._update_slot(domain, slot, state[domain]["semi"][slot])
            # print("update reqt {} in semi".format(slot),
            #       state[domain]["semi"][slot])

    def _update_booked(self, state, domain):
        # check state and goal is fulfill
        self.domain_goals[domain]["booked"] = DEF_VAL_BOOKED
        print("booked")
        for booked_slot in state[domain]["book"]["booked"][0]:
            if self._check_update_request(domain, booked_slot):
                self._update_slot(domain, booked_slot,
                                  state[domain]["book"]["booked"][0][booked_slot])

    def _update_book(self, state, domain, slot):
        if self._check_value(state[domain]["book"][slot]):
            self._update_slot(domain, slot, state[domain]["book"][slot])

    def _check_update_request(self, domain, slot):
        # check whether one slot is a request slot
        if domain not in self.domain_goals:
            return False
        if 'reqt' not in self.domain_goals[domain]:
            return False
        if slot not in self.domain_goals[domain]['reqt']:
            return False
        return True

    def _check_value(self, value=None):
        if not value:
            return False
        if value in NOT_SURE_VALS:
            return False
        return True

    def _get_booking_domain(self, slot, value):
        """ 
        find the domain for domain booking, excluding slot "ref"
        """
        found = ""
        if not slot:  # work around
            return found
        slot = slot.lower()
        value = value.lower()
        for domain in self.all_values["all_value"]:
            if slot in self.all_values["all_value"][domain]:
                if value in self.all_values["all_value"][domain][slot]:
                    if domain in self.domains:
                        found = domain
        return found

    def _init_domain_id(self, domain):
        self.local_id[domain] = {"ID": [0] * self.max_domain_len, "SLOT": {}}

    def _init_slot_id(self, domain, slot):
        self.local_id[domain]["SLOT"][slot] = [0] * self.max_slot_len

    def _update_domain_id(self, domain, domain_id):
        if domain_id < self.max_domain_len:
            self.local_id[domain]["ID"][domain_id] = 1
        else:
            print(
                f"too many doamins: {domain_id} > {self.max_domain_len}")

    def _update_slot_id(self, domain, slot, slot_id):
        if slot_id < self.max_slot_len:
            self.local_id[domain]["SLOT"][slot][slot_id] = 1
        else:
            print(
                f"too many slots, {slot_id} > {self.max_slot_len}")
