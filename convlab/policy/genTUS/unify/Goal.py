"""
The user goal for unify data format
"""
import json


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

    def __init__(self, goal_generator=None, goal=None):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        self.domains = []
        self.domain_goals = {}
        self.status = {}

        if not goal_generator and not goal:
            print("Warning!!! One of goal_generator or goal should not be None!!!")

        if goal_generator:
            print("we only support goal from dataset currently")
            # TODO support multiwpz goal generator

        if goal:
            self._init_goal_from_data(goal)

        self._init_status()

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def _init_goal_from_data(self, goal):
        for intent, domain, slot, value in goal:
            if domain not in self.domains:
                self.domains.append(domain)
                self.domain_goals[domain] = {}
            if intent not in self.domain_goals[domain]:
                self.domain_goals[domain][intent] = {}

            # if slot in self.domain_goals[domain][intent]:
            #     print(
            #         f"duplicate slot!! {intent}-{domain}-{slot}-{self.domain_goals[domain][intent][slot]}/{value}")

            if not value:
                if intent == "request":
                    self.domain_goals[domain][intent][slot] = DEF_VAL_UNK
                else:
                    print(
                        f"unknown no value intent {domain}, {intent}, {slot}")
            else:
                self.domain_goals[domain][intent][slot] = value

    def _init_status(self):
        for domain in self.domain_goals:
            if domain not in self.status:
                self.status[domain] = {}
            for slot_type in self.domain_goals[domain]:
                if slot_type not in self.status[domain]:
                    self.status[domain][slot_type] = {}
                for slot in self.domain_goals[domain][slot_type]:
                    if slot not in self.status[domain][slot_type]:
                        self.status[domain][slot_type][slot] = {}
                    self.status[domain][slot_type][slot] = {
                        "value": str(self.domain_goals[domain][slot_type][slot]),
                        "status": NOT_MENTIONED}

    def get_goal_list(self, data_goal=None):
        if data_goal:
            goal_list = []
            # make sure the order!!!
            for intent, domain, slot, _ in data_goal:
                status = self.status[domain][intent][slot]["status"]
                value = self.domain_goals[domain][intent][slot]
                goal_list.append([intent, domain, slot, value, status])
            return goal_list
        else:
            print("only user history goal is supported currently")

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        pass

    def update_user_goal(self, action, char="usr"):
        # update request and booked
        if char == "usr":
            self._user_action_update(action)
        elif char == "sys":
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
            goal_intent = self._check_slot_and_intent(domain, slot)
            if not goal_intent:
                continue
            # fulfill request by system
            if is_inform(intent) and is_request(goal_intent):
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
        old_value = self.domain_goals[domain][intent][slot]
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


def is_inform(intent):
    if "inform" in intent:
        return True
    return False


def is_request(intent):
    if "request" in intent:
        return True
    return False


def transform_data_act(data_action):
    action_list = []
    for action_type in data_action:
        for act in data_action[action_type]:
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
    data_name = "data/unified_datasets_goal/data.json"
    data = json.load(open(data_name))
    i = 0
    for dialog in data:
        goal = Goal(goal=dialog["unified_goal"])
        print(goal)

        for turn in dialog["turns"]:
            if turn["speaker"] == "user":
                act = transform_data_act(turn["dialogue_act"])
                goal.update_user_goal(action=act, char="usr")
                print("---> usr")
                print(act)
                print(goal)
            if turn["speaker"] == "system":
                act = transform_data_act(turn["dialogue_act"])
                goal.update_user_goal(action=act, char="sys")
                print("---> sys")
                print(act)
                print(goal)

        i += 1
        if i > 0:
            break
