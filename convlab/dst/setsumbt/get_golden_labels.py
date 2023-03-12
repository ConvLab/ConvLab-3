import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

from tqdm import tqdm

from convlab.util import load_dataset
from convlab.util import load_dst_data
from convlab.dst.setsumbt.dataset.value_maps import VALUE_MAP, DOMAINS_MAP, QUANTITIES, TIME


def extract_data(dataset_names: str) -> list:
    dataset_dicts = [load_dataset(dataset_name=name) for name in dataset_names.split('+')]
    data = []
    for dataset_dict in dataset_dicts:
        dataset = load_dst_data(dataset_dict, data_split='test', speaker='all', dialogue_acts=True, split_to_turn=False)
        for dial in dataset['test']:
            data.append(dial)

    return data

def clean_state(state):
    clean_state = dict()
    for domain, subset in state.items():
        clean_state[domain] = {}
        for slot, value in subset.items():
            # Remove pipe separated values
            value = value.split('|')

            # Map values using value_map
            for old, new in VALUE_MAP.items():
                value = [val.replace(old, new) for val in value]
            value = '|'.join(value)

            # Map dontcare to "do not care" and empty to 'none'
            value = value.replace('dontcare', 'do not care')
            value = value if value else 'none'

            # Map quantity values to the integer quantity value
            if 'people' in slot or 'duration' in slot or 'stay' in slot:
                try:
                    if value not in ['do not care', 'none']:
                        value = int(value)
                        value = str(value) if value < 10 else QUANTITIES[-1]
                except:
                    value = value
            # Map time values to the most appropriate value in the standard time set
            elif 'time' in slot or 'leave' in slot or 'arrive' in slot:
                try:
                    if value not in ['do not care', 'none']:
                        # Strip after/before from time value
                        value = value.replace('after ', '').replace('before ', '')
                        # Extract hours and minutes from different possible formats
                        if ':' not in value and len(value) == 4:
                            h, m = value[:2], value[2:]
                        elif len(value) == 1:
                            h = int(value)
                            m = 0
                        elif 'pm' in value:
                            h = int(value.replace('pm', '')) + 12
                            m = 0
                        elif 'am' in value:
                            h = int(value.replace('pm', ''))
                            m = 0
                        elif ':' in value:
                            h, m = value.split(':')
                        elif ';' in value:
                            h, m = value.split(';')
                        # Map to closest 5 minutes
                        if int(m) % 5 != 0:
                            m = round(int(m) / 5) * 5
                            h = int(h)
                            if m == 60:
                                m = 0
                                h += 1
                            if h >= 24:
                                h -= 24
                        # Set in standard 24 hour format
                        h, m = int(h), int(m)
                        value = '%02i:%02i' % (h, m)
                except:
                    value = value
            # Map boolean slots to yes/no value
            elif 'parking' in slot or 'internet' in slot:
                if value not in ['do not care', 'none']:
                    if value == 'free':
                        value = 'yes'
                    elif True in [v in value.lower() for v in ['yes', 'no']]:
                        value = [v for v in ['yes', 'no'] if v in value][0]

            value = value if value != 'none' else ''

            clean_state[domain][slot] = value

    return clean_state

def extract_states(data):
    states_data = {}
    for dial in data:
        states = []
        for turn in dial['turns']:
            if 'state' in turn:
                state = clean_state(turn['state'])
                states.append(state)
        states_data[dial['dialogue_id']] = states

    return states_data


def get_golden_state(prediction, data):
    state = data[prediction['dial_idx']][prediction['utt_idx']]
    pred = prediction['predictions']['state']
    pred = {domain: {slot: pred.get(DOMAINS_MAP.get(domain, domain.lower()), dict()).get(slot, '')
                     for slot in state[domain]} for domain in state}
    prediction['state'] = state
    prediction['predictions']['state'] = pred

    return prediction


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, help='Name of dataset', default="multiwoz21")
    parser.add_argument('--model_path', type=str, help='Path to model dir')
    args = parser.parse_args()

    data = extract_data(args.dataset_name)
    data = extract_states(data)

    reader = open(os.path.join(args.model_path, "predictions", "test.json"), 'r')
    predictions = json.load(reader)
    reader.close()

    predictions = [get_golden_state(pred, data) for pred in tqdm(predictions)]

    writer = open(os.path.join(args.model_path, "predictions", f"test_{args.dataset_name}.json"), 'w')
    json.dump(predictions, writer)
    writer.close()
