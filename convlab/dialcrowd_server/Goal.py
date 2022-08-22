import json
import os

from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.policy.tus.multiwoz.Da2Goal import SysDa2Goal, UsrDa2Goal
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA

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

# Goal slot mapping table
mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'price': 'pricerange'},
           'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name',
                     'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
           'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'type': 'type'},
           'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination',
                     'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
           'taxi': {'car': 'car type', 'phone': 'phone'},
           'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
           'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}

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

    def __init__(self, goal):
        self.domain_goals = _process_goal(goal)
        self.domains = [d for d in self.domain_goals]

        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(path, 'data/multiwoz/all_value.json')
        self.all_values = json.load(open(path))

        self.init_info_record()
        self.actions = None
        self.evaluator = MultiWozEvaluator()
        self.evaluator.add_goal(self.domain_goals)
        self.cur_domain = None

    def init_info_record(self):
        self.info = {}
        for domain in self.domains:
            if 'info' in self.domain_goals[domain].keys():
                self.info[domain] = {}
                for slot in self.domain_goals[domain]['info']:
                    self.info[domain][slot] = DEF_VAL_NUL

    def add_sys_da(self, sys_act, belief_state):
        self.evaluator.add_sys_da(sys_act, belief_state)
        self.update_user_goal(sys_act, belief_state)

    def add_usr_da(self, usr_act):
        self.evaluator.add_usr_da(usr_act)

        usr_domain = [d for i, d, s, v in usr_act][0] if usr_act else self.cur_domain
        usr_domain = usr_domain if usr_domain else 'general'
        self.cur_domain = usr_domain if usr_domain.lower() not in ['general', 'booking'] else self.cur_domain

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        if self.evaluator.success == 1:
            return True
        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain]:
                reqt_vals = self.domain_goals[domain]['reqt'].values()
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        return False
            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return False
        return True

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

    def get_booking_domain(self, slot, value, all_values):
        for domain in self.domains:
            if slot in all_values["all_value"] and value in all_values["all_value"][slot]:
                return domain
        print("NOT FOUND BOOKING DOMAIN")
        return ""

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
        # need to check info, reqt for booked?
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
                    # print(
                    #     f"booking info is incorrect, for slot {slot}: "
                    #     f"goal {user_value} != state {state_value}")
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

            if intent.lower() == 'inform':
                if domain.lower() in self.domain_goals:
                    if 'reqt' in self.domain_goals[domain.lower()]:
                        if REF_SYS_DA_M.get(domain, {}).get(slot, slot) in self.domain_goals[domain]['reqt']:
                            if value in NOT_SURE_VALS:
                                value = '\"' + value + '\"'
                            self.domain_goals[domain]['reqt'][REF_SYS_DA_M.get(domain, {}).get(slot, slot)] = value

            if domain not in ['general', 'booking']:
                self.cur_domain = domain

            if domain and intent and slot:
                dial_act = f'{domain.lower()}-{intent.lower()}-{slot.lower()}'
            else:
                dial_act = ''

            if dial_act == 'booking-book-ref' and self.cur_domain.lower() in ['hotel', 'restaurant', 'train']:
                if self.cur_domain in self.domain_goals and 'booked' in self.domain_goals[self.cur_domain.lower()]:
                    self.domain_goals[self.cur_domain.lower()]['booked'] = DEF_VAL_BOOKED
            elif dial_act == 'train-offerbooked-ref' or dial_act == 'train-inform-ref':
                if 'train' in self.domain_goals and 'booked' in self.domain_goals['train']:
                    self.domain_goals['train']['booked'] = DEF_VAL_BOOKED
            elif dial_act == 'taxi-inform-car':
                if 'taxi' in self.domain_goals and 'booked' in self.domain_goals['taxi']:
                    self.domain_goals['taxi']['booked'] = DEF_VAL_BOOKED
            if intent.lower() in ['book', 'offerbooked'] and self.cur_domain.lower() in self.domain_goals:
                if 'booked' in self.domain_goals[self.cur_domain.lower()]:
                    self.domain_goals[self.cur_domain.lower()]['booked'] = DEF_VAL_BOOKED

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
            # else:
            #     print(
            #         f"UNSEEN SLOT IN UPDATE GOAL {intent, domain, slot, value}")
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
                # print("update reqt {} in booked".format(booked_slot),
                #       self.domain_goals[domain]['reqt'][booked_slot])

    def _update_book(self, state, domain, slot):
        if self._check_value(state[domain]["book"][slot]):
            self._update_slot(domain, slot, state[domain]["book"][slot])
            # print("update reqt {} in book".format(slot),
            #       state[domain]["book"][slot])

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


def _process_goal(tasks):
    goal = {}
    for task in tasks['tasks']:
        goal[task['Dom'].lower()] = {}
        if task['Book']:
            goal[task['Dom'].lower()]['booked'] = DEF_VAL_UNK
            goal[task['Dom'].lower()]['book'] = {}
            for con in task['Book'].split(', '):
                slot, val = con.split('=', 1)
                slot = mapping[task['Dom'].lower()].get(slot, slot)
                goal[task['Dom'].lower()]['book'][slot] = val
        if task['Cons']:
            goal[task['Dom'].lower()]['info'] = {}
            goal[task['Dom'].lower()]['fail_info'] = {}
            for con in task['Cons'].split(', '):
                slot, val = con.split('=', 1)
                slot = mapping[task['Dom'].lower()].get(slot, slot)
                if " (otherwise " in val:
                    value = val.split(" (if unavailable use: ")
                    goal[task['Dom'].lower()]['fail_info'][slot] = value[0]
                    goal[task['Dom'].lower()]['info'][slot] = value[1][:-1]
                else:
                    goal[task['Dom'].lower()]['info'][slot] = val

        if task['Reqs']:
            goal[task['Dom'].lower()]['reqt'] = {mapping[task['Dom'].lower()].get(s, s): DEF_VAL_UNK for s in
                                                 task['Reqs'].split(', ')}

    return goal