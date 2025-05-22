import json
from pprint import pprint
from copy import deepcopy

from tqdm import tqdm
from convlab.util import load_dataset
from convlab.dst.setsumbt.tracker import SetSUMBTTracker


tracker = SetSUMBTTracker(model_name_or_path='ConvLab/setsumbt-dst-multiwoz21')

dataset_name = 'multiwoz21'
dataset = load_dataset(dataset_name)

out = {}
for dialog in tqdm(dataset['test']):
    dialog_id = dialog['original_id'].lower().replace('.json', '')
    out[dialog_id] = []
    states = []

    tracker.init_session()
    turns = dialog['turns']
    prev_u = turns[0]['utterance']
    state = deepcopy(tracker.update(prev_u))
    out[dialog_id].append({'state': state['belief_state']})

    for s, u in zip(turns[1::2], turns[2::2]):
        usr = u['utterance']
        sys = s['utterance']
        
        tracker.state['history'].append(['usr', prev_u])
        tracker.state['history'].append(['sys', sys])
        state = deepcopy(tracker.update(usr))
        out[dialog_id].append({'state': state['belief_state']})
        prev_u = usr

with open('predictions.json', 'w') as f:
    json.dump(out, f, indent=4)
