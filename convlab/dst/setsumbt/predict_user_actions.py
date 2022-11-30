# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Predict dataset user action using SetSUMBT Model"""

from copy import deepcopy
import os
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from convlab.util.custom_util import flatten_acts as flatten
from convlab.util import load_dataset, load_policy_data
from convlab.dst.setsumbt import SetSUMBTTracker


def flatten_acts(acts: dict) -> list:
    """
    Flatten dictionary actions.

    Args:
        acts: Dictionary acts

    Returns:
        flat_acts: Flattened actions
    """
    acts = flatten(acts)
    flat_acts = []
    for intent, domain, slot, value in acts:
        flat_acts.append([intent,
                          domain,
                          slot if slot != 'none' else '',
                          value.lower() if value != 'none' else ''])

    return flat_acts


def get_user_actions(context: list, system_acts: list) -> list:
    """
    Extract user actions from the data.

    Args:
        context: Previous dialogue turns.
        system_acts: List of flattened system actions.

    Returns:
        user_acts: List of flattened user actions.
    """
    user_acts = context[-1]['dialogue_acts']
    user_acts = flatten_acts(user_acts)
    if len(context) == 3:
        prev_state = context[-3]['state']
        cur_state = context[-1]['state']
        for domain, substate in cur_state.items():
            for slot, value in substate.items():
                if prev_state[domain][slot] != value:
                    act = ['inform', domain, slot, value]
                    if act not in user_acts and act not in system_acts:
                        user_acts.append(act)

    return user_acts


def extract_dataset(dataset: str = 'multiwoz21') -> list:
    """
    Extract acts and utterances from the dataset.

    Args:
        dataset: Dataset name

    Returns:
        data: Extracted data
    """
    data = load_dataset(dataset_name=dataset)
    raw_data = load_policy_data(data, data_split='test', context_window_size=3)['test']

    dialogue = list()
    data = list()
    for turn in raw_data:
        state = dict()
        state['system_utterance'] = turn['context'][-2]['utterance'] if len(turn['context']) > 1 else ''
        state['utterance'] = turn['context'][-1]['utterance']
        state['system_actions'] = turn['context'][-2]['dialogue_acts'] if len(turn['context']) > 1 else {}
        state['system_actions'] = flatten_acts(state['system_actions'])
        state['user_actions'] = get_user_actions(turn['context'], state['system_actions'])
        dialogue.append(state)
        if turn['terminated']:
            data.append(dialogue)
            dialogue = list()

    return data


def unflatten_acts(acts: list) -> dict:
    """
    Convert acts from flat list format to dict format.

    Args:
        acts: List of flat actions.

    Returns:
        unflat_acts: Dictionary of acts.
    """
    binary_acts = []
    cat_acts = []
    for intent, domain, slot, value in acts:
        include = True if (domain == 'general') or (slot != 'none') else False
        if include and (value == '' or value == 'none' or intent == 'request'):
            binary_acts.append({'intent': intent,
                                'domain': domain,
                                'slot': slot if slot != 'none' else ''})
        elif include:
            cat_acts.append({'intent': intent,
                             'domain': domain,
                             'slot': slot if slot != 'none' else '',
                             'value': value})

    unflat_acts = {'categorical': cat_acts, 'binary': binary_acts, 'non-categorical': list()}

    return unflat_acts


def predict_user_acts(data: list, tracker: SetSUMBTTracker) -> list:
    """
    Predict the user actions using the SetSUMBT Tracker.

    Args:
        data: List of dialogues.
        tracker: SetSUMBT Tracker

    Returns:
        predict_result: List of turns containing predictions and true user actions.
    """
    tracker.init_session()
    predict_result = []
    for dial_idx, dialogue in enumerate(data):
        for turn_idx, state in enumerate(dialogue):
            sample = {'dial_idx': dial_idx, 'turn_idx': turn_idx}

            tracker.state['history'].append(['sys', state['system_utterance']])
            predicted_state = deepcopy(tracker.update(state['utterance']))
            tracker.state['history'].append(['usr', state['utterance']])
            tracker.state['system_action'] = state['system_actions']

            sample['predictions'] = {'dialogue_acts': unflatten_acts(predicted_state['user_action'])}
            sample['dialogue_acts'] = unflatten_acts(state['user_actions'])

            predict_result.append(sample)

        tracker.init_session()

    return predict_result


if __name__ =="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, help='Name of dataset', default="multiwoz21")
    parser.add_argument('--model_path', type=str, help='Path to model dir')
    args = parser.parse_args()

    dataset = extract_dataset(args.dataset_name)
    tracker = SetSUMBTTracker(args.model_path)
    predict_results = predict_user_acts(dataset, tracker)

    with open(os.path.join(args.model_path, 'predictions', 'test_nlu.json'), 'w') as writer:
        json.dump(predict_results, writer, indent=2)
        writer.close()
