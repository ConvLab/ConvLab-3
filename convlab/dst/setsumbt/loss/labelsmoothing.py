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
"""Label smoothing loss function"""


import torch
from torch.nn import Softmax, Module, CrossEntropyLoss
from torch.nn.functional import kl_div


class LabelSmoothingLoss(Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing: float = 0.05, ignore_index: int = -1) -> Module:
        '''
        Args:
            label_smoothing: Label smoothing constant
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(LabelSmoothingLoss, self).__init__()

        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            labels: Label indices

        Returns:
            loss: Loss value
        '''
        # Assert input sizes
        assert input.dim() == 2
        assert labels.dim() == 1
        assert self.label_smoothing <= ((input.size(-1) - 1) / input.size(-1))

        # Confirm predictive distribution dimension
        if labels.max() <= input.size(-1):
            dimension = input.size(-1)
        else:
            raise NameError(f'Label dimension {labels.max()} is larger than prediction dimension {input.size(-1)}.')

        # Remove observations to be ignored in loss calculation
        input = input[labels != self.ignore_index]
        labels = labels[labels != self.ignore_index]

        # Create target distribution
        input = torch.log(torch.softmax(input, -1))
        targets = torch.ones(input.size()).float().to(input.device)
        targets *= self.label_smoothing / (dimension - 1)
        targets[range(labels.size(0)), labels] = 1.0 - self.label_smoothing

        return kl_div(input, targets, reduction='none').sum(-1).mean()


class BinaryLabelSmoothingLoss(LabelSmoothingLoss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing: float = 0.05, ignore_index: int = -1) -> Module:
        '''
        Args:
            label_smoothing: Label smoothing constant
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(LabelSmoothingLoss, self).__init__(label_smoothing, ignore_index)

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            labels: Label indices

        Returns:
            loss: Loss value
        '''
        # Assert input sizes
        assert input.dim() == 1
        assert labels.dim() == 1
        assert self.label_smoothing <= 0.5

        input = torch.sigmoid(input).reshape(-1, 1)
        input = torch.log(torch.cat((1 - input, input), 1))
        targets = torch.ones(input.size()).float().to(input.device)
        targets *= self.label_smoothing
        targets[range(labels.size(0)), labels.long()] = 1.0 - self.label_smoothing

        return kl_div(input, targets, reduction='none').sum(-1).mean()
