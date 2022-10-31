# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Inhibited Softmax Activation and Loss Functions"""


import torch
from torch.nn import Softmax, Module, CrossEntropyLoss
from torch.nn.functional import kl_div


class LabelSmoothingLoss(Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing=0.05, ignore_index=-1):
        super(LabelSmoothingLoss, self).__init__()

        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        assert logits.dim() == 2
        assert targets.dim() == 1
        assert self.label_smoothing <= ((logits.size(-1) - 1) / logits.size(-1))

        logits = logits[targets != self.ignore_index]
        targets = targets[targets != self.ignore_index]

        logits = torch.log(torch.softmax(logits, -1))
        labels = torch.ones(logits.size()).float().to(logits.device)
        labels *= self.label_smoothing / (logits.size(-1) - 1)
        labels[range(labels.size(0)), targets] = 1.0 - self.label_smoothing

        kl = kl_div(logits, labels, reduction='none').sum(-1).mean()
        del logits, targets, labels
        return kl


class BinaryLabelSmoothingLoss(Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing=0.05):
        super(BinaryLabelSmoothingLoss, self).__init__()

        assert 0.0 < label_smoothing <= 1.0
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        assert logits.dim() == 1
        assert targets.dim() == 1
        assert self.label_smoothing <= 0.5

        logits = torch.sigmoid(logits).reshape(-1, 1)
        logits = torch.log(torch.cat((1 - logits, logits), 1))
        labels = torch.ones(logits.size()).float().to(logits.device)
        labels *= self.label_smoothing
        labels[range(labels.size(0)), targets.long()] = 1.0 - self.label_smoothing

        kl = kl_div(logits, labels, reduction='none').sum(-1).mean()
        del logits, targets
        return kl
