"""
The user goal for unify data format
"""
import json
from convlab.policy.tus.unify.Goal import old_goal2list
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator

from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal as ABUS_Goal
from convlab.policy.tus.unify.Goal import Goal as TUS_Goal
from convlab.util.custom_util import slot_mapping
DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, ""]

NOT_MENTIONED = "not mentioned"
FULFILLED = "fulfilled"
REQUESTED = "requested"
CONFLICT = "conflict"


class Goal:
    """ User Goal Model Class. """

    def __init__(self, goal=None, goal_generator=None):
        """
        create new Goal from a dialog or from goal_generator
        Args:
            goal: can be a list (create from a dialog), an abus goal, or none
        """
        self.domains = []
        self.domain_goals = {}
        self.status = {}
        self.invert_slot_mapping = {v: k for k, v in slot_mapping.items()}
        self.raw_goal = None
        self._init_goal_from_data(goal, goal_generator)
        self._init_status()
        self.evaluator = MultiWozEvaluator(check_book_constraints=False)
        if self.raw_goal is None:
            self.raw_goal = self.domain_goals
        self.evaluator.add_goal(self.raw_goal)

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def _old_goal(self, goal=None, goal_generator=None):
        if not goal and goal_generator:
            goal = ABUS_Goal(goal_generator)
            self.raw_goal = goal.domain_goals
            goal = old_goal2list(goal.domain_goals)

        elif isinstance(goal, dict):
            self.raw_goal = goal
            goal = old_goal2list(goal)

        elif isinstance(goal, ABUS_Goal) or isinstance(goal, TUS_Goal):
            self.raw_goal = goal.domain_goals
            goal = old_goal2list(goal.domain_goals)

        # else:
        #     print("unknow goal")
        return goal

    def _init_goal_from_data(self, goal=None, goal_generator=None):
        goal = self._old_goal(goal, goal_generator)

        # be careful of this order
        for domain, intent, slot, value in goal:
            if domain == "none":
                continue
            if domain not in self.domains:
                self.domains.append(domain)
                self.domain_goals[domain] = {}
            if intent not in self.domain_goals[domain]:
                self.domain_goals[domain][intent] = {}

            if not value:
                if intent == "request":
                    self.domain_goals[domain][intent][slot] = DEF_VAL_UNK
                else:
                    print(
                        f"unknown no value intent {domain}, {intent}, {slot}")
            else:
                self.domain_goals[domain][intent][slot] = value

    def _init_status(self):
        for domain, domain_goal in self.domain_goals.items():
            if domain not in self.status:
                self.status[domain] = {}
            for slot_type, sub_domain_goal in domain_goal.items():
                if slot_type not in self.status[domain]:
                    self.status[domain][slot_type] = {}
                for slot in sub_domain_goal:
                    if slot not in self.status[domain][slot_type]:
                        self.status[domain][slot_type][slot] = {}
                    self.status[domain][slot_type][slot] = {
                        "value": str(sub_domain_goal[slot]),
                        "status": NOT_MENTIONED}

    def get_goal_list(self, data_goal=None, sub_goal_success=False):
        goal_list = []
        if data_goal:
            # make sure the order!!!
            for domain, intent, slot, _ in data_goal:
                status = self._get_status(domain, intent, slot)
                value = self.domain_goals[domain][intent][slot]
                goal_list.append([intent, domain, slot, value, status])
            return goal_list
        else:
            add_next_domain = True
            for domain, domain_goal in self.domain_goals.items():
                if not add_next_domain:
                    return goal_list

                for intent, sub_goal in domain_goal.items():
                    for slot, value in sub_goal.items():
                        status = self._get_status(domain, intent, slot)
                        goal_list.append([intent, domain, slot, value, status])
                if sub_goal_success:
                    add_next_domain = self.sub_goal_success(
                        domain, domain_goal)

        return goal_list

    def _get_status(self, domain, intent, slot):
        if domain not in self.status:
            return NOT_MENTIONED
        if intent not in self.status[domain]:
            return NOT_MENTIONED
        if slot not in self.status[domain][intent]:
            return NOT_MENTIONED
        return self.status[domain][intent][slot]["status"]

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        if self.evaluator.task_success():
            return True
        return False

        for domain, domain_goal in self.domain_goals.items():
            if not self.sub_goal_success(domain, domain_goal):
                return False

        # for domain, domain_goal in self.status.items():
        #     if domain not in self.domain_goals:
        #         continue
        #     for slot_type, sub_domain_goal in domain_goal.items():
        #         if slot_type not in self.domain_goals[domain]:
        #             continue
        #         for slot, status in sub_domain_goal.items():
        #             if slot not in self.domain_goals[domain][slot_type]:
        #                 continue
        #             # for strict success, turn this on
        #             if status["status"] in [NOT_MENTIONED, CONFLICT]:
        #                 if status["status"] == CONFLICT and slot in ["arrive by", "leave at"]:
        #                     continue
        #                 return False
        #             if "?" in status["value"]:
        #                 return False

        return True

    def sub_goal_success(self, domain, goal):
        for intent, sub_goal in goal.items():
            for slot in sub_goal:
                if "?" in self.status[domain][intent][slot]["value"]:
                    return False
                status = self.status[domain][intent][slot]["status"]
                if status in [NOT_MENTIONED, CONFLICT]:
                    if status == CONFLICT and slot in ["arrive by", "leave at"]:
                        continue
                    return False

        return True

    # TODO change to update()?
    def update_user_goal(self, action, char="usr"):
        # update request and booked
        if char == "usr":
            self.evaluator.add_usr_da(action)
            self._user_action_update(action)
        elif char == "sys":
            self.evaluator.add_sys_da(action)
            self._system_action_update(action)
        else:
            print("!!!UNKNOWN CHAR!!!")

    def _user_action_update(self, action):
        # no need to update user goal
        for intent, domain, slot, _ in action:
            goal_intent = self._check_slot_and_intent(domain, slot)
            if not goal_intent:
                continue
            # fulfilled by user
            if is_inform(intent):
                self._set_status(goal_intent, domain, slot, FULFILLED)
            # requested by user
            if is_request(intent):
                self._set_status(goal_intent, domain, slot, REQUESTED)

    def _system_action_update(self, action):
        for intent, domain, slot, value in action:
            value = missing_value_in_binary_action(intent, domain, slot, value)
            goal_intent = self._check_slot_and_intent(domain, slot)
            if not goal_intent:
                continue
            # fulfill request by system
            if is_inform(intent) and is_request(goal_intent):
                if value in ["not available", "none"]:
                    continue
                self._set_status(goal_intent, domain, slot, FULFILLED)
                self._set_goal(goal_intent, domain, slot, value)

            if is_inform(intent) and is_inform(goal_intent):
                # fulfill infrom by system
                if value == self.domain_goals[domain][goal_intent][slot]:
                    self._set_status(goal_intent, domain, slot, FULFILLED)
                # conflict system inform
                else:
                    self._set_status(goal_intent, domain, slot, CONFLICT)
            # requested by system
            if is_request(intent) and is_inform(goal_intent):
                self._set_status(goal_intent, domain, slot, REQUESTED)

    def _set_status(self, intent, domain, slot, status):
        self.status[domain][intent][slot]["status"] = status

    def _set_goal(self, intent, domain, slot, value):
        # old_value = self.domain_goals[domain][intent][slot]
        self.domain_goals[domain][intent][slot] = value
        self.status[domain][intent][slot]["value"] = value
        # print(
        #     f"updating user goal {intent}-{domain}-{slot} {old_value}-> {value}")

    def _check_slot_and_intent(self, domain, slot):
        not_found = ""
        if domain not in self.domain_goals:
            return not_found
        for intent in self.domain_goals[domain]:
            if slot in self.domain_goals[domain][intent]:
                return intent
        return not_found


def missing_value_in_binary_action(intent, domain, slot, value):
    # a workaround for missing slot value in binary action, e.g. hotel-inform(parking)
    if len(value) < 1:
        if intent == "inform" and domain is not None and slot is not None:
            return "yes"
        return None
    return value


def is_inform(intent):
    if "inform" in intent:
        return True
    if "recommend" in intent:
        return True
    # if "select" in intent:
    #     return True
    # if "offerbook" in intent:
    #     return True
    return False


def is_request(intent):
    if "request" in intent:
        return True
    return False


def transform_data_act(data_action):
    action_list = []
    for _, dialog_act in data_action.items():
        for act in dialog_act:
            value = act.get("value", "")
            if not value:
                if "request" in act["intent"]:
                    value = "?"
                else:
                    value = "none"
            action_list.append(
                [act["intent"], act["domain"], act["slot"], value])
    return action_list


if __name__ == "__main__":
    from pprint import pprint
    from copy import deepcopy
    goal_generator = GoalGenerator()
    goal = Goal(goal_generator=goal_generator)
    new_goal = deepcopy(goal)
    new_goal.update_user_goal(
        [["inform", "restaurant", "food", "asian oriental"]], char="sys")
    pprint(goal.get_goal_list())
    pprint(new_goal.get_goal_list())
