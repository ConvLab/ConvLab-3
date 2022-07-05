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
"""KL Divergence Ensemble Distillation loss"""

import torch
from torch import lgamma, log
from torch.nn import Module
from torch.nn.functional import kl_div


# Pytorch BayesianMatchingLoss nn.Module (see https://arxiv.org/pdf/1503.02531.pdf for details)
class KLDistillationLoss(Module):

    def __init__(self, lamb: float = 1e-4, ignore_index: int = -1) -> Module:
        '''
        Ensemble Distillation loss using KL Divergence (https://arxiv.org/pdf/1503.02531.pdf) implementation

        Args:
            lamb: Target smoothing parameter
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(KLDistillationLoss, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, targets: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            targets: Target distribution (ensemble marginal)
            temp: Temperature scaling coefficient for predictive distribution

        Returns:
            loss: Loss value
        '''
        # Assert input sizes
        assert input.dim() == 2                 # Observations, predictive distribution
        assert targets.dim() == 2                # Label for each observation
        assert targets.size(0) == input.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if targets.size(-1) == input.size(-1):
            dimension = input.size(-1)
        else:
            raise NameError(f'Target dimension {targets.size(-1)} is not the same as the prediction dimension {input.size(-1)}.')

        # Remove observations to be ignored in loss calculation
        input = torch.log(torch.softmax(input / temp, -1))
        ids = torch.where(targets[:, 0] != self.ignore_index)[0]
        input = input[ids]
        targets = targets[ids]

        # Target smoothing
        targets = ((1 - self.lamb) * targets) + (self.lamb / targets.size(-1))

        return kl_div(input, targets, reduction='none').sum(-1).mean()


# Pytorch BayesianMatchingLoss nn.Module
class BinaryKLDistillationLoss(KLDistillationLoss):

    def __init__(self, lamb: float = 1e-4, ignore_index: int = -1) -> Module:
        '''
        Binary Ensemble Distillation loss using KL Divergence (https://arxiv.org/pdf/1503.02531.pdf) implementation

        Args:
            lamb: Target smoothing parameter
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(BinaryKLDistillationLoss, self).__init__(lamb, ignore_index)

    def forward(self, input: torch.Tensor, targets: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            targets: Target distribution (ensemble marginal)
            temp: Temperature scaling coefficient for predictive distribution

        Returns:
            loss: Loss value
        '''
        # Assert input sizes
        assert input.dim() == 1                 # Observations, predictive distribution
        assert targets.dim() == 1                # Label for each observation
        assert targets.size(0) == input.size(0)  # Equal number of observation
        
        # Convert input and target to 2D binary distribution for KL divergence computation
        input = torch.sigmoid(input / temp).unsqueeze(-1)
        input = torch.log(torch.cat((1 - input, input), 1))

        targets = targets.unsqueeze(-1)
        targets = torch.cat((1 - targets, targets), -1)

        return super().forward(input, targets, temp)
