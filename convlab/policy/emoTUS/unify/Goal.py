"""
The user goal for unify data format
"""
from convlab.policy.genTUS.unify.Goal import Goal as GenTUSGoal
from convlab.policy.genTUS.unify.Goal import DEF_VAL_UNK
from random import random
from convlab.util.custom_util import set_seed


class Goal(GenTUSGoal):
    """ User Goal Model Class. """

    def __init__(self, goal=None, goal_generator=None, use_sentiment=False):
        """
        create new Goal from a dialog or from goal_generator
        Args:
            goal: can be a list (create from a dialog), an abus goal, or none
        """
        super().__init__(goal, goal_generator)
        self.use_sentiment = use_sentiment
        # TODO sample Exciting? User politeness?

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

    def emotion_info(self):
        self.user_persona = {"user": "Polite"}
        event = {}
        z = random()
        if z > 0.95:
            self.user_persona["user"] = "Impolite"
        # TODO: should check domains only in the user goal

        for d in self.domains:
            # Excited
            z = random()
            if z > 0.8 and d in ["restaurant", "attraction", "train"]:
                event[d] = "Excited"
            z = random()
            if z > 0.95 and d in ["restaurant", "police", "hospital"] and d not in event:
                event[d] = "Fearful"

        if event:
            self.user_persona["event"] = event

        return self.user_persona


def emotion_info(dialog=None, goal=None):
    user_persona = {"user": "Polite"}
    event_emotion = {1: "Fearful", 5: "Excited"}
    event = {}
    if dialog is None:
        # politeness
        z = random()
        if z > 0.95:
            user_persona = "Impolite"
        # TODO: should check domains only in the user goal

        for d in ["restaurant", "attraction", "train"]:
            z = random()
            if z > 0.8:
                event[d] = "Excited"
        for d in ["restaurant", "police", "hospital"]:
            if d in event:
                continue
            z = random()
            if z > 0.95:
                event[d] = "Fearful"
        if event:
            user_persona["event"] = event

    else:
        for turn in dialog['turns']:
            if turn['speaker'] == 'user':
                emotion = turn["emotion"][-1]["emotion"]
                # Fearful and Excited
                if int(emotion) in event_emotion:
                    domain = check_domain(turn["dialogue_acts"])
                    for d in domain:
                        if d not in event:
                            event[d] = event_emotion[emotion]
                # Abusive
                if int(emotion) == 4:
                    user_persona["user"] = "Impolite"
        if event:
            user_persona["event"] = event

    return user_persona


def check_domain(dialog_act):
    domain = []
    for _, acts in dialog_act.items():
        for act in acts:
            if act["domain"] == "general":
                continue
            if act["domain"] not in domain:
                domain.append(act["domain"])
    return domain
