import time
import json
from convlab.policy.tus.unify.util import split_slot_name, slot_name_map
from convlab.util.custom_util import slot_mapping

from random import sample, shuffle
from pprint import pprint
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK, ""]


# only support user goal from dataset


def is_time(goal, status):
    if isTimeFormat(goal) and isTimeFormat(status):
        return True
    return False


def isTimeFormat(input):
    try:
        time.strptime(input, '%H:%M')
        return True
    except ValueError:
        return False


def old_goal2list(goal: dict, reorder=False) -> list:
    goal_list = []
    for domain in goal:
        for slot_type in ['info', 'book', 'reqt']:
            if slot_type not in goal[domain]:
                continue
            temp = []
            for slot in goal[domain][slot_type]:
                s = slot
                if slot in slot_name_map:
                    s = slot_name_map[slot]
                elif slot in slot_name_map[domain]:
                    s = slot_name_map[domain][slot]
                # domain, intent, slot, value
                if slot_type in ['info', 'book']:
                    i = "inform"
                    v = goal[domain][slot_type][slot]
                else:
                    i = "request"
                    v = DEF_VAL_UNK
                s = slot_mapping.get(s, s)
                temp.append([domain, i, s, v])
            shuffle(temp)
            goal_list = goal_list + temp
    # shuffle_goal = goal_list[:1] + sample(goal_list[1:], len(goal_list)-1)
    # return shuffle_goal
    return goal_list


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal: list = None):
        """
        create new Goal by random
        Args:
            goal (list): user goal built from user history
            ontology (dict): domains, slots, values
        """
        self.goal = goal
        self.max_domain_len = 6
        self.max_slot_len = 20
        self.local_id = {}

        self.domains = []
        # goal: {domain: {"info": {slot: value}, "reqt": {slot:?}}, ...}
        self.domain_goals = {}
        # status: {domain: {slot: value}}
        self.status = {}
        self.user_history = {}
        self.init_goal_status(goal)
        self.init_local_id()

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def init_goal_status(self, goal):
        for domain, intent, slot, value in goal:  # check this order
            if domain not in self.domains:
                self.domains.append(domain)
                self.domain_goals[domain] = {}

            # "book" domain is not clear for unify data format

            if "request" in intent.lower():
                if "reqt" not in self.domain_goals[domain]:
                    self.domain_goals[domain]["reqt"] = {}
                self.domain_goals[domain]["reqt"][slot] = DEF_VAL_UNK

            elif "info" in intent.lower():
                if "info" not in self.domain_goals[domain]:
                    self.domain_goals[domain]["info"] = {}
                self.domain_goals[domain]["info"][slot] = value

            self.user_history[f"{domain}-{slot}"] = value

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domain_goals:
            if domain not in self.status:
                # print(f"{domain} is not mentioned")
                return False
            if "info" in self.domain_goals[domain]:
                for slot in self.domain_goals[domain]["info"]:
                    if slot not in self.status[domain]:
                        # print(f"{slot} is not mentioned")
                        return False
                    goal = self.domain_goals[domain]["info"][slot].lower()
                    status = self.status[domain][slot].lower()
                    if goal != status and not is_time(goal, status):
                        # print(f"conflict slot {slot}: {goal} <-> {status}")
                        return False
            if "reqt" in self.domain_goals[domain]:
                for slot in self.domain_goals[domain]["reqt"]:
                    if self.domain_goals[domain]["reqt"][slot] == DEF_VAL_UNK:
                        # print(f"not fulfilled request{domain}-{slot}")
                        return False

        return True

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
        # domain, slot = new_slot_name.split('-')
        domain, slot = split_slot_name(new_slot_name)
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
        # print(slot_name)
        # domain, slot = slot_name.split('-')
        domain, slot = split_slot_name(slot_name)
        if domain in self.local_id and slot in self.local_id[domain]["SLOT"]:
            return self.local_id[domain]["ID"], self.local_id[domain]["SLOT"][slot]
        else:  # a slot not in original user goal
            self.insert_local_id(slot_name)
            domain_id, slot_id = self.get_slot_id(slot_name)
            return domain_id, slot_id

    def action_list(self, sys_act=None):
        priority_action = [x for x in self.user_history]

        if sys_act:
            for _, domain, slot, _ in sys_act:
                slot_name = f"{domain}-{slot}"
                if slot_name and slot_name not in priority_action:
                    priority_action.insert(0, slot_name)

        return priority_action

    def update(self, action: list = None, char: str = "system"):
        # update request and booked
        if char not in ["user", "system"]:
            print(f"unknown role: {char}")
        self._update_status(action, char)
        self._update_goal(action, char)
        return self.status

    def _update_status(self, action: list, char: str):
        for intent, domain, slot, value in action:
            if slot == "arrive by":
                slot = "arriveBy"
            elif slot == "leave at":
                slot = "leaveAt"
            if domain not in self.status:
                self.status[domain] = {}
            # update info
            if "info" in intent:
                self.status[domain][slot] = value
            elif "request" in intent:
                self.status[domain][slot] = DEF_VAL_UNK

    def _update_goal(self, action: list, char: str):
        # update requt slots in goal
        for intent, domain, slot, value in action:
            if slot == "arrive by":
                slot = "arriveBy"
            elif slot == "leave at":
                slot = "leaveAt"
            if "info" not in intent:
                continue
            if self._check_update_request(domain, slot) and value != "?":
                self.domain_goals[domain]['reqt'][slot] = value
                # print(f"update reqt {slot} = {value} from system action")

    def _update_slot(self, domain, slot, value):
        self.domain_goals[domain]['reqt'][slot] = value

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


if __name__ == "__main__":
    data_goal = [["restaurant", "inform", "cuisine", "punjabi"],
                 ["restaurant", "inform", "city", "milpitas"],
                 ["restaurant", "request", "price_range", ""],
                 ["restaurant", "request", "street_address", ""]]
    goal = Goal(data_goal)
    print(goal)
    action = {"char": "system",
              "action": [["request", "restaurant", "cuisine", "?"], ["request", "restaurant", "city", "?"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "user",
              "action": [["inform", "restaurant", "cuisine", "punjabi"], ["inform", "restaurant", "city", "milpitas"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "system",
              "action": [["inform", "restaurant", "price_range", "cheap"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "user",
              "action": [["request", "restaurant", "street_address", ""]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
    action = {"char": "system",
              "action": [["inform", "restaurant", "street_address", "ABCD"]]}
    goal.update(action["action"], action["char"])
    print(goal.status)
    print("complete:", goal.task_complete())
