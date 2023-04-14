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
"""Loss functions for SetSUMBT"""

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from convlab.dst.setsumbt.modeling.loss.bayesian_matching import (BayesianMatchingLoss,
                                                                  BinaryBayesianMatchingLoss)
from convlab.dst.setsumbt.modeling.loss.kl_distillation import KLDistillationLoss, BinaryKLDistillationLoss
from convlab.dst.setsumbt.modeling.loss.labelsmoothing import LabelSmoothingLoss, BinaryLabelSmoothingLoss
from convlab.dst.setsumbt.modeling.loss.endd_loss import (RKLDirichletMediatorLoss,
                                                          BinaryRKLDirichletMediatorLoss)

LOSS_MAP = {
    'crossentropy': {'non-binary': CrossEntropyLoss,
                     'binary': BCEWithLogitsLoss,
                     'args': list()},
    'bayesianmatching': {'non-binary': BayesianMatchingLoss,
                         'binary': BinaryBayesianMatchingLoss,
                         'args': ['kl_scaling_factor']},
    'labelsmoothing': {'non-binary': LabelSmoothingLoss,
                       'binary': BinaryLabelSmoothingLoss,
                       'args': ['label_smoothing']},
    'distillation': {'non-binary': KLDistillationLoss,
                     'binary': BinaryKLDistillationLoss,
                     'args': ['ensemble_smoothing']},
    'distribution_distillation': {'non-binary': RKLDirichletMediatorLoss,
                                  'binary': BinaryRKLDirichletMediatorLoss,
                                  'args': []}
}

def load(loss_function, binary=False):
    """
    Load loss function

    Args:
        loss_function (str): Loss function name
        binary (bool): Whether to use binary loss function

    Returns:
        torch.nn.Module: Loss function
    """
    assert loss_function in LOSS_MAP
    args_list = LOSS_MAP[loss_function]['args']
    loss_function = LOSS_MAP[loss_function]['binary' if binary else 'non-binary']

    def __init__(ignore_index=-1, **kwargs):
        args = {'ignore_index': ignore_index} if loss_function != BCEWithLogitsLoss else dict()
        for arg, val in kwargs.items():
            if arg in args_list:
                args[arg] = val

        return loss_function(**args)

    return __init__
