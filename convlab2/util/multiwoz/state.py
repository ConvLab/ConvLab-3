def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={},
                 booked={},
                 request_state={},
                 terminated=False,
                 history=[])
    return state
