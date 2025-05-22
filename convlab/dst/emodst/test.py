from convlab.dst.emodst.tracker import EMODST 

setsumbt_args = { 
        'model_name_or_path': '/Users/shutong/Projects/EmoLoop/required_files/dst/setsumbt-dst_nlu-multiwoz21-EnD2' # path to the setsumbt repository
    }

trippy_args = {
    'model_path': 'ConvLab/roberta-base-trippy-dst-multiwoz21'   # path to the trippy repository on huggingface
}

tracker = EMODST(
    kwargs_for_erc={
        'base_model_type': 'bert-base-uncased',
        'base_model_path': '/Users/shutong/Projects/EmoLoop/required_files/dst/bert-base-uncased',   # NEW: path to the BERT model that saved locally
        'model_type': 'contextbert-ertod',
        'model_name_or_path': '/Users/shutong/Projects/EmoLoop/required_files/dst/contextbert-ertod.pt'  # path to the contextbert checkpoint
    },
    dst_model_name='setsumbt',
    kwargs_for_dst=setsumbt_args
)

tracker.init_session()

# # prepending empty strings required by trippy
# tracker.dst.state['history'].append(['usr', ''])
# tracker.dst.state['history'].append(['sys', ''])
user_act = 'hey. I need a cheap restaurant.'
state = tracker.update(user_act)
print(state)

tracker.dst.state['history'].append(['usr', 'hey. I need a cheap restaurant.'])
tracker.dst.state['history'].append(['sys', 'There are many cheap places, which food do you like?'])
user_act = 'If you have something Asian that would be great.'
state = tracker.update(user_act)
print(state)

tracker.dst.state['history'].append(['usr', 'If you have something Asian that would be great.'])
tracker.dst.state['history'].append(['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
tracker.dst.state['system_action'] = [['inform', 'restaurant', 'food', 'chinese'],
                                    ['inform', 'restaurant', 'name', 'the golden wok']]
user_act = 'Great!'
state = tracker.update(user_act)
print(state)

# tracker.state['history'].append(['usr', 'Great. Where are they located?'])
# state = tracker.state
# state['terminated'] = False
# state['booked'] = {}

