def act2str(act):
    """Convert unified dataset dialog act dict to string.
    act:
        {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'area', 'value': 'north'}],
        'non-categorical': [{'intent': 'inform', 'domain': 'hotel', 'slot': 'area', 'value': 'north'}],
        'binary': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}]}
    return:
        restaurant-inform(area=north) hotel-inform(area=north) hotel-request(area=?)
    """
    old_format_dict = convert2old_format(act)
    return dict2seq(old_format_dict)


def convert2old_format(act):
    """
    dict: {'categorical': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}], 'non-categorical': [...], 'binary': [,,,]}
    """
    new_act = {}
    for key in act:
        for item_dic in act[key]:
            domain = item_dic['domain']
            if domain not in new_act:
                new_act[domain] = {}
            intent = item_dic['intent']
            if intent not in new_act[domain]:
                new_act[domain][intent] = []
            slot = item_dic['slot']
            if 'value' in item_dic:
                value = item_dic['value']
            else:
                value = None
            new_act[domain][intent].append([slot, value])
    return new_act


def dict2seq(d):
    '''
    dict: [domain: { intent: [slot, value] }]
    seq: [domain { intent ( slot = value ; ) @ } | ]
    '''
    s = ''
    for domain in d:
        for intent in d[domain]:
            if 'book' in intent.lower():
                domain_str = 'booking'
                intent_str = 'book'
            else:
                domain_str = domain
                intent_str = intent
            domain_action_str = f'{domain_str}-{intent_str}'
            s += f'{domain_action_str}('
            inner_strs = []
            for slot, value in d[domain][intent]:
                slot_str = str(slot).lower()
                value_str = str(value).lower()
                if domain_action_str == 'booking-book':
                    if slot_str == 'none' and value_str != 'none':
                        slot_str = 'ref'
                # if domain_action_str == 'booking-book':
                #     if slot_str == 'none':
                #         continue
                if slot_str == 'price range':
                    slot_str = 'price'

                if (slot_str == 'none' and value_str == 'none'):
                    inner_strs.append(f' ')
                elif (slot_str != 'none' and value_str == 'none'):
                    inner_strs.append(f'{slot_str}=?')
                elif (slot_str == 'none' and value_str != 'none'):
                    inner_strs.append(f'{value_str}')
                else:
                    inner_strs.append(f'{slot_str}={value_str}')
            s += ','.join(inner_strs)
            s += ') '
    # return s.lower()
    return s

if __name__ == '__main__':
    ipt = {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'area', 'value': 'north'}], 'non-categorical': [{'intent': 'inform', 'domain': 'hotel', 'slot': 'area', 'value': 'north'}], 'binary': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}]}
    print(act2str(ipt))
