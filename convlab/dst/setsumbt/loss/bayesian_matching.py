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
"""Bayesian Matching Activation and Loss Functions (see https://arxiv.org/pdf/2002.07965.pdf for details)"""

import torch
from torch import digamma, lgamma
from torch.nn import Module


# Pytorch BayesianMatchingLoss nn.Module
class BayesianMatchingLoss(Module):

    def __init__(self, lamb: float = 0.001, ignore_index: int = -1) -> Module:
        '''
        Bayesian matching loss (https://arxiv.org/pdf/2002.07965.pdf) implementation

        Args:
            lamb: Weighting factor for the KL Divergence component
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(BayesianMatchingLoss, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            labels: Label indices
            prior: Prior distribution over label classes

        Returns:
            loss: Loss value
        '''
        # Assert input sizes
        assert input.dim() == 2                 # Observations, predictive distribution
        assert labels.dim() == 1                # Label for each observation
        assert labels.size(0) == alpha.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if labels.max() <= input.size(-1):
            dimension = input.size(-1)
        else:
            raise NameError(f'Label dimension {labels.max()} is larger than prediction dimension {input.size(-1)}.')
        
        # Remove observations to be ignored in loss calculation
        if prior is not None:
            prior = prior[labels != self.ignore_index]
        input = torch.exp(input[labels != self.ignore_index])
        labels = labels[labels != self.ignore_index]
        
        # Initialise and reshape prior parameters
        if prior is None:
            prior = torch.ones(dimension).to(alpha.device)
        prior = prior.to(alpha.device)

        # KL divergence term (divergence of predictive distribution from prior over label classes - regularisation term)
        log_gamma_term = lgamma(input.sum(-1)) - lgamma(prior.sum(-1)) + (lgamma(prior) - lgamma(input)).sum(-1)
        div_term = digamma(input) - digamma(input.sum(-1)).unsqueeze(-1).repeat((1, input.size(-1)))
        div_term = ((input - prior) * div_term).sum(-1)
        kl_term = log_gamma_term + div_term
        kl_term *= self.lamb
        del log_gamma_term, div_term, prior

        # Expected log likelihood
        expected_likelihood = digamma(input[range(labels.size(0)), labels]) - digamma(input.sum(-1))
        del input, labels

        # Apply ELBO loss and mean reduction
        loss = (kl - expected_likelihood).mean()
        del kl, expected_likelihood

        return loss


# Pytorch BayesianMatchingLoss nn.Module for Binary classification
class BinaryBayesianMatchingLoss(BayesianMatchingLoss):

    def __init__(self, lamb: float = 0.001, ignore_index: int = -1) -> Module:
        '''
        Bayesian matching loss (https://arxiv.org/pdf/2002.07965.pdf) implementation

        Args:
            lamb: Weighting factor for the KL Divergence component
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        '''
        super(BinaryBayesianMatchingLoss, self).__init__(lamb, ignore_index)

    def forward(self, input: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
        '''
        Args:
            input: Predictive distribution
            labels: Label indices
            prior: Prior distribution over label classes

        Returns:
            loss: Loss value
        '''
        
        # Create 2D input dirichlet distribution
        input_sum = 1 + (1 / self.lamb)
        input = (torch.sigmoid(input) * input_sum).reshape(-1, 1)
        input = torch.cat((input_sum - input, input), 1)

        return super().forward(input, labels, prior=prior)
