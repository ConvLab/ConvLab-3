def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={
                     'attraction': {'type': '', 'name': '', 'area': ''}, 
                     'hotel': {'name': '', 'area': '', 'parking': '', 'price range': '', 'stars': '4', 'internet': 'yes', 'type': 'hotel', 'book stay': '', 'book day': '', 'book people': ''}, 
                     'restaurant': {'food': '', 'price range': '', 'name': '', 'area': '', 'book time': '', 'book day': '', 'book people': ''}, 
                     'taxi': {'leave at': '', 'destination': '', 'departure': '', 'arrive by': ''}, 
                     'train': {'leave at': '', 'destination': '', 'day': '', 'arrive by': '', 'departure': '', 'book people': ''}, 
                     'hospital': {'department': ''}
                     },
                 booked={},
                 request_state={},
                 terminated=False,
                 history=[])
    return state


def default_state_old():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={},
                 request_state={},
                 terminated=False,
                 history=[])
    state['belief_state'] = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    }
    return state
