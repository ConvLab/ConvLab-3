import json
import os

from copy import deepcopy
from convlab.util import load_ontology
from convlab.util.multiwoz.state import default_state
from convlab.dst.rule.multiwoz.dst_util import normalize_value
from convlab.dst.dst import DST


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``convlab.util.multiwoz.state.default_state`` returns a default state.
        value_dict(dict):
            It helps check whether ``user_act`` has correct content.
    """

    def __init__(self, dataset_name='multiwoz21'):
        DST.__init__(self)
        self.ontology = load_ontology(dataset_name)
        self.state = default_state()
        self.default_belief_state = deepcopy(self.ontology['state'])
        self.state['belief_state'] = deepcopy(self.default_belief_state)
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

    def update(self, user_act=None):
        """
        update belief_state, request_state
        :param user_act:
        :return:
        """
        for intent, domain, slot, value in user_act:
            if domain not in self.state['belief_state']:
                continue
            if intent == 'inform':
                if slot == 'none' or slot == '':
                    continue
                domain_dic = self.state['belief_state'][domain]
                if slot in domain_dic:
                    nvalue = normalize_value(
                        self.value_dict, domain, slot, value)
                    self.state['belief_state'][domain][slot] = nvalue
                elif slot != 'none' or slot != '':
                    # raise Exception('unknown slot name <{}> of domain <{}>'.format(k, domain))
                    with open('unknown_slot.log', 'a+') as f:
                        f.write(
                            'unknown slot name <{}> of domain <{}>\n'.format(slot, domain))
            elif intent == 'request':
                if domain not in self.state['request_state']:
                    self.state['request_state'][domain] = {}
                if slot not in self.state['request_state'][domain]:
                    self.state['request_state'][domain][slot] = 0
        # self.state['user_action'] = user_act  # should be added outside DST module
        return self.state

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``convlab.util.multiwoz.state.default_state`` returns."""
        self.state = default_state()
        self.state['belief_state'] = deepcopy(self.default_belief_state)


if __name__ == '__main__':
    # from convlab.dst.rule.multiwoz import RuleDST

    dst = RuleDST()

    # Action (dialog acts) is a list of (intent, domain, slot, value) tuples.
    # RuleDST will only handle `inform` and `request` actions
    action = [
        ["inform", "hotel", "area", "east"],
        ["inform", "hotel", "stars", "4"]
    ]

    # method `update` updates the attribute `state` of tracker, and returns it.
    state = dst.update(action)
    assert state == dst.state
    assert state == {'belief_state': {'attraction': {'area': '', 'name': '', 'type': ''},
                                      'hospital': {'department': ''},
                                      'hotel': {'area': 'east',
                                                'book day': '',
                                                'book people': '',
                                                'book stay': '',
                                                'internet': '',
                                                'name': '',
                                                'parking': '',
                                                'price range': '',
                                                'stars': '4',
                                                'type': ''},
                                      'restaurant': {'area': '',
                                                     'book day': '',
                                                     'book people': '',
                                                     'book time': '',
                                                     'food': '',
                                                     'name': '',
                                                     'price range': ''},
                                      'taxi': {'arrive by': '',
                                               'departure': '',
                                               'destination': '',
                                               'leave at': ''},
                                      'train': {'arrive by': '',
                                                'book people': '',
                                                'day': '',
                                                'departure': '',
                                                'destination': '',
                                                'leave at': ''}},
                     'booked': {},
                     'history': [],
                     'request_state': {},
                     'system_action': [],
                     'terminated': False,
                     'user_action': []}

    # Please call `init_session` before a new dialog. This initializes the attribute `state` of tracker with a default state, which `convlab.util.multiwoz.state.default_state` returns. But You needn't call it before the first dialog, because tracker gets a default state in its constructor.
    dst.init_session()
    action = [["inform", "train", "arrive by", "19:45"]]
    state = dst.update(action)
    assert state == {'belief_state': {'attraction': {'area': '', 'name': '', 'type': ''},
                                      'hospital': {'department': ''},
                                      'hotel': {'area': '',
                                                'book day': '',
                                                'book people': '',
                                                'book stay': '',
                                                'internet': '',
                                                'name': '',
                                                'parking': '',
                                                'price range': '',
                                                'stars': '',
                                                'type': ''},
                                      'restaurant': {'area': '',
                                                     'book day': '',
                                                     'book people': '',
                                                     'book time': '',
                                                     'food': '',
                                                     'name': '',
                                                     'price range': ''},
                                      'taxi': {'arrive by': '',
                                               'departure': '',
                                               'destination': '',
                                               'leave at': ''},
                                      'train': {'arrive by': '19:45',
                                                'book people': '',
                                                'day': '',
                                                'departure': '',
                                                'destination': '',
                                                'leave at': ''}},
                     'booked': {},
                     'history': [],
                     'request_state': {},
                     'system_action': [],
                     'terminated': False,
                     'user_action': []}
