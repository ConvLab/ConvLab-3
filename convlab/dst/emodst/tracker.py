# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Shutong Feng (fengs@hhu.de)
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
"""Run SetSUMBT belief tracker training and evaluation."""

import copy
import logging

import torch
import transformers

import json
from convlab.dst.emodst.modeling.emotion_estimator import EmotionEstimator

# dst prerequisit
from convlab.dst.dst import DST
# for supported dst
from convlab.dst.setsumbt.tracker import SetSUMBTTracker
from convlab.dst.trippy.tracker import TRIPPY
# for supported nlu-ruledst pipeline
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU

# list of todos:
# currently only support trippy and setsubmt (or single DST). Expand to NLU-DST scenarios

SUPPORTED_DST = {
    'setsumbt': SetSUMBTTracker,
    'trippy': TRIPPY
}

SUPPORTED_NLU = {
    'bertnlu': BERTNLU
}

transformers.logging.set_verbosity_error()


class NLU_DST(DST):
    def __init__(self,
                 nlu_model_name: str = 'bert',
                 kwargs_for_nlu: dict = {}):
        super(NLU_DST, self).__init__()
        self.nlu = SUPPORTED_NLU[nlu_model_name](**kwargs_for_nlu)
        self.dst = RuleDST()

    def init_session(self):
        self.dst.init_session()
        self.state = self.dst.state

    def update(self, user_act: str = '') -> dict:
        sem_act = self.nlu.predict(user_act, context=self.state['history'])
        self.state = self.dst.update(sem_act)
        return self.state


class EMODST(DST):
    """ERC object combined with DST for Convlab dialogue system"""

    def __init__(self,
                 dst_model_name: str = 'setsumbt',
                 kwargs_for_erc: dict = {},
                 kwargs_for_dst: dict = {}):

        super(EMODST, self).__init__()

        self.erc = EmotionEstimator(kwargs_for_erc)

        # if use dst
        if dst_model_name in SUPPORTED_DST:
            self.dst = SUPPORTED_DST[dst_model_name](**kwargs_for_dst)
        # if use nlu-ruledst pipeline
        elif dst_model_name in SUPPORTED_NLU:
            self.dst = NLU_DST(nlu_model_name=dst_model_name,
                               kwargs_for_nlu=kwargs_for_dst)
        else:
            raise NameError('DSTNotImplemented')

        self.emotion2id = json.load(
            open('convlab/dst/emodst/modeling/emotion2id.json'))
        self.id2emotion = {v: k for k, v in self.emotion2id.items()}

    def init_session(self):
        self.dst.init_session()
        self.dialog_state_history = []
        self.state = self.dst.state
        self.state['user_emotion'] = None
        self.state['user_emotion_trajectory'] = []

    def update(self, user_act: str = '') -> dict:
        """
        Update dialogue state based on user utterance.

        Args:
            user_act: User utterance, or actions for RuleDST

        Returns:
            state: Dialogue state
        """
        # question: sort out state history update?
        # it seems like the history has been updated somewhere

        self.state = self.dst.update(user_act)
        self.dialog_state_history.append(copy.deepcopy(self.state))

        emotion = self.erc.predict(
            user_utt=user_act,
            dialog_state_history=self.dialog_state_history
        )
        emotion = emotion.to('cpu').item()

        self.state['user_emotion_trajectory'].append(emotion)
        self.state['user_emotion'] = emotion

        return self.state

    def get_emotion(self):
        return self.id2emotion[self.state['user_emotion']]
