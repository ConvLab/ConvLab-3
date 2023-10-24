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


from convlab.dst.emodst.modeling.emotion_estimator import EmotionEstimator
from convlab.dst.dst import DST
# supported dst
from convlab.dst.setsumbt.tracker import SetSUMBTTracker
from convlab.dst.trippy.tracker import TRIPPY

# list of todos:
# currently only support trippy and setsubmt (or single DST). Expand to NLU-DST scenarios

SUPPORTED_DST = {
    'setsumbt': SetSUMBTTracker,
    'trippy': TRIPPY
}

transformers.logging.set_verbosity_error()

class EMODST(DST):
    """ERC object combined with DST for Convlab dialogue system"""

    def __init__(self,
                 dst_model_name: str = 'setsumbt',
                 kwargs_for_erc: dict = {},
                 kwargs_for_dst: dict = {}):
        
        super(EMODST, self).__init__()

        self.erc = EmotionEstimator(kwargs_for_erc)
        
        if dst_model_name in SUPPORTED_DST:
            self.dst = SUPPORTED_DST[dst_model_name](**kwargs_for_dst)
        else:
            raise NameError('DSTNotImplemented')

    def init_session(self):
        self.dst.init_session()
        # self.erc.init_session()
        self.dialog_state_history = []
        self.state = self.dst.state
        self.state['user_emotion'] = None
        self.state['user_emotion_trajectory'] = []


    def update(self, user_act: str = '') -> dict:
        """
        Update dialogue state based on user utterance.

        Args:
            user_act: User utterance

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
