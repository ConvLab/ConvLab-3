import json
import os

from convlab.util.multiwoz.state import default_state_old as default_state
from convlab.dst.rule.multiwoz.dst_util import normalize_value
from convlab.dst.rule.multiwoz import RuleDST
from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from convlab.policy.tus.multiwoz.Da2Goal import SysDa2Goal, UsrDa2Goal
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import unified_format, act_dict_to_flat_tuple
from pprint import pprint
from copy import deepcopy
from convlab.util import relative_import_module_from_unified_datasets

reverse_da = relative_import_module_from_unified_datasets('multiwoz21', 'preprocess.py', 'reverse_da')

SLOT2SEMI = {
    "arriveby": "arriveBy",
    "leaveat": "leaveAt",
    "trainid": "trainID",
}


class UserRuleDST(RuleDST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``convlab.util.multiwoz.state.default_state`` returns a default state.
        value_dict(dict):
            It helps check whether ``user_act`` has correct content.
    """

    def __init__(self, dataset_name='multiwoz21'):
        super().__init__()

        self.state = default_state()
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))
        self.mentioned_domain = []

    def update(self, sys_act=None):
        """
        update belief_state, request_state
        :param sys_act:
        :return:
        """
        sys_act = unified_format(sys_act)
        sys_act = reverse_da(sys_act)
        sys_act = act_dict_to_flat_tuple(sys_act)
        # print("dst", user_act)
        self.update_mentioned_domain(sys_act)
        for intent, domain, slot, value in sys_act:
            domain = domain.lower()
            intent = intent.lower()
            if domain in ['unk', 'general']:
                continue
            # TODO domain: booking
            if domain == "booking":
                for domain in self.mentioned_domain:
                    self.update_inform_request(
                        intent, domain, slot, value)
            else:
                self.update_inform_request(intent, domain, slot, value)
        return self.state

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``convlab.util.multiwoz.state.default_state`` returns."""
        self.state = default_state()
        self.mentioned_domain = []

    def update_mentioned_domain(self, sys_act):
        if not sys_act:
            return
        for intent, domain, slot, value in sys_act:
            domain = domain.lower()
            if domain not in self.mentioned_domain and domain not in ['unk', 'general', 'booking']:
                self.mentioned_domain.append(domain)
                # print(f"update: mentioned {domain} domain")

    def update_inform_request(self, intent, domain, slot, value):
        slot = slot.lower()
        k = SysDa2Goal[domain].get(slot, slot)
        k = SLOT2SEMI.get(k, k)
        if k is None:
            return
        try:
            assert domain in self.state['belief_state']
        except:
            raise Exception(
                f'Error: domain <{domain}> not in new belief state')
        domain_dic = self.state['belief_state'][domain]
        assert 'semi' in domain_dic
        assert 'book' in domain_dic
        if k in domain_dic['semi']:
            nvalue = normalize_value(self.value_dict, domain, k, value)
            self.state['belief_state'][domain]['semi'][k] = nvalue
        elif k in domain_dic['book']:
            self.state['belief_state'][domain]['book'][k] = value
        elif k.lower() in domain_dic['book']:
            self.state['belief_state'][domain]['book'][k.lower()
                                                       ] = value
        elif k == 'trainID' and domain == 'train':
            self.state['belief_state'][domain]['book'][k] = normalize_value(
                self.value_dict, domain, k, value)
        else:
            # print('unknown slot name <{}> of domain <{}>'.format(k, domain))
            nvalue = normalize_value(self.value_dict, domain, k, value)
            self.state['belief_state'][domain]['semi'][k] = nvalue
            with open('unknown_slot.log', 'a+') as f:
                f.write(
                    'unknown slot name <{}> of domain <{}>\n'.format(k, domain))

    def update_request(self):
        pass

    def update_booking(self):
        pass


if __name__ == '__main__':
    # from convlab.dst.rule.multiwoz import RuleDST

    dst = UserRuleDST()

    action = [['Inform', 'Restaurant', 'Phone', '01223323737'],
              ['reqmore', 'general', 'none', 'none'],
              ["Inform", "Hotel", "Area", "east"], ]
    state = dst.update(action)
    pprint(state)
    dst.init_session()

    # Action is a dict. Its keys are strings(domain-type pairs, both uppercase and lowercase is OK) and its values are list of lists.
    # The domain may be one of ('Attraction', 'Hospital', 'Booking', 'Hotel', 'Restaurant', 'Taxi', 'Train', 'Police').
    # The type may be "inform" or "request".

    # For example, the action below has a key "Hotel-Inform", in which "Hotel" is domain and "Inform" is action type.
    # Each list in the value of "Hotel-Inform" is a slot-value pair. "Area" is slot and "east" is value. "Star" is slot and "4" is value.
    action = [
        ["Inform", "Hotel", "Area", "east"],
        ["Inform", "Hotel", "Stars", "4"]
    ]

    # method `update` updates the attribute `state` of tracker, and returns it.
    state = dst.update(action)
    assert state == dst.state
    assert state == {'user_action': [],
                     'system_action': [],
                     'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                                      'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                                'semi': {'name': '',
                                                         'area': 'east',
                                                         'parking': '',
                                                         'pricerange': '',
                                                         'stars': '4',
                                                         'internet': '',
                                                         'type': ''}},
                                      'attraction': {'book': {'booked': []},
                                                     'semi': {'type': '', 'name': '', 'area': ''}},
                                      'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                                     'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                                      'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                                      'taxi': {'book': {'booked': []},
                                               'semi': {'leaveAt': '',
                                                        'destination': '',
                                                        'departure': '',
                                                        'arriveBy': ''}},
                                      'train': {'book': {'booked': [], 'people': ''},
                                                'semi': {'leaveAt': '',
                                                         'destination': '',
                                                         'day': '',
                                                         'arriveBy': '',
                                                         'departure': ''}}},
                     'request_state': {},
                     'terminated': False,
                     'history': []}

    # Please call `init_session` before a new dialog. This initializes the attribute `state` of tracker with a default state, which `convlab.util.multiwoz.state.default_state` returns. But You needn't call it before the first dialog, because tracker gets a default state in its constructor.
    dst.init_session()
    action = [["Inform", "Train", "Arrive", "19:45"]]
    state = dst.update(action)
    assert state == {'user_action': [],
                     'system_action': [],
                     'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                                      'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                                'semi': {'name': '',
                                                         'area': '',
                                                         'parking': '',
                                                         'pricerange': '',
                                                         'stars': '',
                                                         'internet': '',
                                                         'type': ''}},
                                      'attraction': {'book': {'booked': []},
                                                     'semi': {'type': '', 'name': '', 'area': ''}},
                                      'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                                     'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                                      'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                                      'taxi': {'book': {'booked': []},
                                               'semi': {'leaveAt': '',
                                                        'destination': '',
                                                        'departure': '',
                                                        'arriveBy': ''}},
                                      'train': {'book': {'booked': [], 'people': ''},
                                                'semi': {'leaveAt': '',
                                                         'destination': '',
                                                         'day': '',
                                                         'arriveBy': '19:45',
                                                         'departure': ''}}},
                     'request_state': {},
                     'terminated': False,
                     'history': []}
