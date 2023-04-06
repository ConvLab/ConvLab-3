# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Ensemble setup ad inference utils."""

import os
from shutil import copy2 as copy

import torch
import numpy as np

def setup_ensemble(model_path: str, ensemble_size: int):
    """
    Setup ensemble model directory structure.

    Args:
        model_path: Path to ensemble model directory
        ensemble_size: Number of ensemble members
    """
    for i in range(ensemble_size):
        path = os.path.join(model_path, f'ens-{i}')
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'dataloaders'))
            # Add development set dataloader to each ensemble member directory
            for set_type in ['dev']:
                copy(os.path.join(model_path, 'dataloaders', f'{set_type}.dataloader'),
                     os.path.join(path, 'dataloaders', f'{set_type}.dataloader'))


class EnsembleAggregator:
    """Aggregator for ensemble model outputs."""

    def __init__(self):
        self.init_session()
        self.input_items = ['input_ids', 'attention_mask', 'token_type_ids']
        self.output_items = ['belief_state', 'request_probabilities', 'active_domain_probabilities',
                             'general_act_probabilities']

    def init_session(self):
        """Initialize aggregator for new session."""
        self.features = dict()

    def add_batch(self, model_input: dict, model_output: dict, dialogue_ids=None):
        """
        Add batch of model outputs to aggregator.

        Args:
            model_input: Model input dictionary
            model_output: Model output dictionary
            dialogue_ids: List of dialogue ids
        """
        for key in self.input_items:
            if key in model_input:
                if key not in self.features:
                    self.features[key] = list()
                self.features[key].append(model_input[key])

        for key in self.output_items:
            if key in model_output:
                if key not in self.features:
                    self.features[key] = list()
                self.features[key].append(model_output[key])

        if dialogue_ids is not None:
            if 'dialogue_ids' not in self.features:
                self.features['dialogue_ids'] = [np.array([list(itm) for itm in dialogue_ids]).T]
            else:
                self.features['dialogue_ids'].append(np.array([list(itm) for itm in dialogue_ids]).T)

    def _aggregate(self):
        """Aggregate model outputs."""
        for key in self.features:
            self.features[key] = self._aggregate_item(self.features[key])

    @staticmethod
    def _aggregate_item(item):
        """
        Aggregate single model output item.

        Args:
            item: Model output item

        Returns:
            Aggregated model output item
        """
        if item[0] is None:
            return None
        elif type(item[0]) == dict:
            return {k: EnsembleAggregator._aggregate_item([i[k] for i in item]) for k in item[0]}
        elif type(item[0]) == np.ndarray:
            return np.concatenate(item, 0)
        else:
            return torch.cat(item, 0)

    def save(self, path):
        """
        Save aggregated model outputs to file.

        Args:
            path: Path to save file
        """
        self._aggregate()
        torch.save(self.features, path)
