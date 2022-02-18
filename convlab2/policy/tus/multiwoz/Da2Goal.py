UsrDa2Goal = {
    'attraction': {
        'area': 'area', 'name': 'name', 'type': 'type',
        'addr': 'address', 'fee': 'entrance fee', 'phone': 'phone',
        'post': 'postcode', 'ref': "ref", 'none': None
    },
    'hospital': {
        'department': 'department', 'addr': 'address', 'phone': 'phone',
        'post': 'postcode', 'ref': "ref", 'none': None
    },
    'hotel': {
        'area': 'area', 'internet': 'internet', 'name': 'name',
        'parking': 'parking', 'price': 'pricerange', 'stars': 'stars',
        'type': 'type', 'addr': 'address', 'phone': 'phone',
        'post': 'postcode', 'day': 'day', 'people': 'people',
        'stay': 'stay', 'ref': "ref", 'none': None
    },
    'police': {
        'addr': 'address', 'phone': 'phone', 'post': 'postcode', 'name': 'name', 'ref': "ref", 'none': None
    },
    'restaurant': {
        'area': 'area', 'day': 'day', 'food': 'food', 'type': 'type',
        'name': 'name', 'people': 'people', 'price': 'pricerange',
        'time': 'time', 'addr': 'address', 'phone': 'phone',
        'post': 'postcode', 'ref': "ref", 'none': None
    },
    'taxi': {
        'arrive': 'arriveby', 'depart': 'departure', 'dest': 'destination',
        'leave': 'leaveat', 'car': 'car type', 'phone': 'phone', 'ref': "ref", 'none': None
    },
    'train': {
        'time': "duration", 'arrive': 'arriveby', 'day': 'day', 'ref': "ref",
        'depart': 'departure', 'dest': 'destination', 'leave': 'leaveat',
        'people': 'people', 'duration': 'duration', 'price': 'price', 'choice': "choice",
        'trainid': 'trainid', 'ticket': 'price', 'id': "trainid", 'none': None
    }
}

SysDa2Goal = {
    'attraction': {
        'addr': "address", 'area': "area", 'choice': "choice",
        'fee': "entrance fee", 'name': "name", 'phone': "phone",
        'post': "postcode", 'price': "pricerange", 'type': "type",
        'none': None
    },
    'booking': {
        'day': 'day', 'name': 'name', 'people': 'people',
        'ref': 'ref', 'stay': 'stay', 'time': 'time',
        'none': None
    },
    'hospital': {
        'department': 'department', 'addr': 'address', 'post': 'postcode',
        'phone': 'phone', 'none': None
    },
    'hotel': {
        'addr': "address", 'area': "area", 'choice': "choice",
        'internet': "internet", 'name': "name", 'parking': "parking",
        'phone': "phone", 'post': "postcode", 'price': "pricerange",
        'ref': "ref", 'stars': "stars", 'type': "type",
        'none': None
    },
    'restaurant': {
        'addr': "address", 'area': "area", 'choice': "choice",
        'name': "name", 'food': "food", 'phone': "phone",
        'post': "postcode", 'price': "pricerange", 'ref': "ref",
        'none': None
    },
    'taxi': {
        'arrive': "arriveby", 'car': "car type", 'depart': "departure",
        'dest': "destination", 'leave': "leaveat", 'phone': "phone",
        'none': None
    },
    'train': {
        'arrive': "arriveby", 'choice': "choice", 'day': "day",
        'depart': "departure", 'dest': "destination", 'id': "trainid", 'trainid': "trainid",
        'leave': "leaveat", 'people': "people", 'ref': "ref",
        'ticket': "price", 'time': "duration", 'duration': 'duration', 'none': None
    },
    'police': {
        'addr': "address", 'post': "postcode", 'phone': "phone", 'name': 'name', 'none': None
    }
}
ref_slot_data2stand = {
    'train': {
        'duration': 'time', 'price': 'ticket', 'trainid': 'id'
    }
}
