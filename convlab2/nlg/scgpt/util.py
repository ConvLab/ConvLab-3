import torch

def act2str(act):
    """Convert unified dataset dialog act dict to string.
    act:
        {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'area', 'value': 'north'}],
        'non-categorical': [{'intent': 'inform', 'domain': 'hotel', 'slot': 'area', 'value': 'north'}],
        'binary': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}]}
    return:
        restaurant { inform ( area = north ) } | hotel { inform ( area = north ) @ request ( area ) }
    """
    old_format_dict = convert2old_format(act)
    return dict2seq(old_format_dict)


def build_mask(max_len, seq_lens, use_float=False):
    """
    make one-hot masks given seq_lens list.
    e.g., input: max_len=4, seq_lens=[2,3], return: [[1,1,0,0], [1,1,1,0]]
    Args:
        max_len (int): maximum sequence length
        seq_lens (torch.Tensor): (batch)
    Returns:
        mask (torch.Tensor): (batch, max_len)
    """
    a = torch.arange(max_len)[None, :]
    b = seq_lens[:, None].cpu()
    mask = a < b
    if use_float:
        mask = mask.float()
    return mask


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
    first_domain = True
    first_intent = True
    first_slot = True
    for domain in d:
        if not first_domain:
            s += ' | '
        s += domain
        s += ' { '
        first_domain = False
        first_intent = True
        for intent in d[domain]:
            if not first_intent:
                s += ' @ '
            s += intent
            s += ' ( '
            first_intent = False
            first_slot = True
            for slot, value in d[domain][intent]:
                if not first_slot:
                    s += ' ; '
                s += slot
                if value:
                    s += ' = '
                    s += value
                first_slot = False
            s += ' )'
        s += ' }'
    return s.lower()


if __name__ == '__main__':
    ipt = {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'area', 'value': 'north'}], 'non-categorical': [{'intent': 'inform', 'domain': 'hotel', 'slot': 'area', 'value': 'north'}], 'binary': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}]}
    print(act2str(ipt))