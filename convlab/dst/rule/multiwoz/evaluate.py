# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
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
"""MultiWOZ Test data inference for RuleDST and BERTNLU+RuleDST"""

import json
from copy import deepcopy
import os

from tqdm import tqdm

from convlab.util import load_dataset, load_dst_data
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU

BERTNLU_PATH = "https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_user_context3.zip"


def flatten_act(acts: dict) -> list:
    acts_list = list()
    for act_type, _acts in acts.items():
        for act in _acts:
            if 'value' in act:
                _act = [act['intent'], act['domain'], act['slot'], act['value']]
            else:
                _act = [act['intent'], act['domain'], act['slot'], '']
            acts_list.append(_act)
    return acts_list


def load_act_data(dataset: dict) -> list:
    data = list()
    for dialogue in tqdm(dataset['test']):
        dial = []
        for _turn in dialogue['turns']:
            if _turn['speaker'] == 'user':
                turn = {'user_acts': flatten_act(_turn['dialogue_acts']),
                        'state': _turn['state']}
                dial.append(turn)
        data.append(dial)
    return data


def load_text_data(dataset: dict) -> list:
    data = list()
    for dialogue in tqdm(dataset['test']):
        dial = []
        turn = {'user': '', 'system': 'Start', 'state': None}
        for _turn in dialogue['turns']:
            if _turn['speaker'] == 'user':
                turn['user'] = _turn['utterance']
                turn['state'] = _turn['state']
            elif _turn['speaker'] == 'system':
                turn['system'] = _turn['utterance']
            if turn['user'] and turn['system']:
                if turn['system'] == 'Start':
                    turn['system'] = ''
                dial.append(deepcopy(turn))
                turn = {'user': '', 'system': '', 'state': None}
        data.append(dial)
    return data


def predict_acts(data: list, nlu: BERTNLU) -> list:
    processed_data = list()
    for dialogue in tqdm(data):
        context = list()
        dial = list()
        for turn in dialogue:
            context.append(['sys', turn['system']])
            acts = nlu.predict(turn['user'], context=context)
            context.append(['usr', turn['user']])
            dial.append({'user_acts': deepcopy(acts), 'state': turn['state']})
        processed_data.append(dial)
    return processed_data


def predict_states(data: list):
    dst = RuleDST()
    processed_data = list()
    for dialogue in tqdm(data):
        dst.init_session()
        for turn in dialogue:
            pred = dst.update(turn['user_acts'])
            dial = {'state': turn['state'],
                    'predictions': {'state': deepcopy(pred['belief_state'])}}
            processed_data.append(dial)
    return processed_data


if __name__ == '__main__':
    dataset = load_dataset(dataset_name='multiwoz21')
    dataset = load_dst_data(dataset, data_split='test', speaker='all', dialogue_acts=True, split_to_turn=False)

    data = load_text_data(dataset)
    nlu = BERTNLU(mode='user', config_file='multiwoz21_user_context3.json', model_file=BERTNLU_PATH)
    bertnlu_data = predict_acts(data, nlu)

    golden_data = load_act_data(dataset)

    bertnlu_data = predict_states(bertnlu_data)
    golden_data = predict_states(golden_data)

    path = os.path.dirname(os.path.realpath(__file__))
    writer = open(os.path.join(path, f"predictions_BERTNLU-RuleDST.json"), 'w')
    json.dump(bertnlu_data, writer)
    writer.close()

    writer = open(os.path.join(path, f"predictions_RuleDST.json"), 'w')
    json.dump(golden_data, writer)
    writer.close()
