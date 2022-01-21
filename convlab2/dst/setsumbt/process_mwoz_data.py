import os
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from tqdm import tqdm

from convlab2.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
from convlab2.util.multiwoz.lexicalize import deflat_da, flat_da


def load_data(path):
    with open(path, 'r') as reader:
        data = json.load(reader)
        reader.close()
    
    return data


def load_tracker(model_checkpoint):
    model = SetSUMBTTracker(model_path=model_checkpoint)
    model.init_session()

    return model


def process_dialogue(dial, model, get_full_belief_state):
    model.store_full_belief_state = get_full_belief_state
    model.init_session()

    model.state['history'].append(['sys', ''])
    processed_dial = []
    belief_state = {}
    for turn in dial:
        if not turn['metadata']:
            state = model.update(turn['text'])
            model.state['history'].append(['usr', turn['text']])
            
            acts = model.state['user_action']
            acts = [[val.replace('-', ' ') for val in act] for act in acts]
            acts = flat_da(acts)
            acts = deflat_da(acts)
            turn['dialog_act'] = acts
        else:
            model.state['history'].append(['sys', turn['text']])
            turn['metadata'] = model.state['belief_state']
        
        if get_full_belief_state:
            for slot, probs in model.full_belief_state.items():
                if slot not in belief_state:
                    belief_state[slot] = [probs[0]]
                else:
                    belief_state[slot].append(probs[0])
        
        processed_dial.append(turn)
    
    if get_full_belief_state:
        belief_state = {slot: torch.cat(probs, 0).cpu() for slot, probs in belief_state.items()}

    return processed_dial, belief_state


def process_dialogues(data, model, get_full_belief_state=False):
    processed_data = {}
    belief_states = {}
    for dial_id, dial in tqdm(data.items()):
        dial['log'], bs = process_dialogue(dial['log'], model, get_full_belief_state)
        processed_data[dial_id] = dial
        if get_full_belief_state:
            belief_states[dial_id] = bs

    return processed_data, belief_states


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_path')
    parser.add_argument('--data_path')
    parser.add_argument('--get_full_belief_state', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    print('Loading data and model...')
    data = load_data(os.path.join(args.data_path, 'data.json'))
    model = load_tracker(args.model_path)

    print('Processing data...\n')
    data, belief_states = process_dialogues(data, model, get_full_belief_state=args.get_full_belief_state)
    
    print('Saving results...\n')
    torch.save(belief_states, os.path.join(args.data_path, 'setsumbt_belief_states.bin'))
    with open(os.path.join(args.data_path, 'setsumbt_data.json'), 'w') as writer:
        json.dump(data, writer, indent=2)
        writer.close()
