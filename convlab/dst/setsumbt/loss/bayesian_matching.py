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


class BayesianMatchingLoss(Module):
    """Bayesian matching loss (https://arxiv.org/pdf/2002.07965.pdf) implementation"""

    def __init__(self, lamb: float = 0.001, ignore_index: int = -1) -> Module:
        """
        Args:
            lamb (float): Weighting factor for the KL Divergence component
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(BayesianMatchingLoss, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Predictive distribution
            labels (Tensor): Label indices
            prior (Tensor): Prior distribution over label classes

        Returns:
            loss (Tensor): Loss value
        """
        # Assert input sizes
        assert inputs.dim() == 2                 # Observations, predictive distribution
        assert labels.dim() == 1                # Label for each observation
        assert labels.size(0) == inputs.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if labels.max() <= inputs.size(-1):
            dimension = inputs.size(-1)
        else:
            raise NameError(f'Label dimension {labels.max()} is larger than prediction dimension {inputs.size(-1)}.')
        
        # Remove observations to be ignored in loss calculation
        if prior is not None:
            prior = prior[labels != self.ignore_index]
        inputs = torch.exp(inputs[labels != self.ignore_index])
        labels = labels[labels != self.ignore_index]
        
        # Initialise and reshape prior parameters
        if prior is None:
            prior = torch.ones(dimension).to(inputs.device)
        prior = prior.to(inputs.device)

        # KL divergence term (divergence of predictive distribution from prior over label classes - regularisation term)
        log_gamma_term = lgamma(inputs.sum(-1)) - lgamma(prior.sum(-1)) + (lgamma(prior) - lgamma(inputs)).sum(-1)
        div_term = digamma(inputs) - digamma(inputs.sum(-1)).unsqueeze(-1).repeat((1, inputs.size(-1)))
        div_term = ((inputs - prior) * div_term).sum(-1)
        kl_term = log_gamma_term + div_term
        kl_term *= self.lamb
        del log_gamma_term, div_term, prior

        # Expected log likelihood
        expected_likelihood = digamma(inputs[range(labels.size(0)), labels]) - digamma(inputs.sum(-1))
        del inputs, labels

        # Apply ELBO loss and mean reduction
        loss = (kl_term - expected_likelihood).mean()
        del kl_term, expected_likelihood

        return loss


class BinaryBayesianMatchingLoss(BayesianMatchingLoss):
    """Bayesian matching loss (https://arxiv.org/pdf/2002.07965.pdf) implementation"""

    def __init__(self, lamb: float = 0.001, ignore_index: int = -1) -> Module:
        """
        Args:
            lamb (float): Weighting factor for the KL Divergence component
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(BinaryBayesianMatchingLoss, self).__init__(lamb, ignore_index)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Predictive distribution
            labels (Tensor): Label indices
            prior (Tensor): Prior distribution over label classes

        Returns:
            loss (Tensor): Loss value
        """
        
        # Create 2D input dirichlet distribution
        input_sum = 1 + (1 / self.lamb)
        inputs = (torch.sigmoid(inputs) * input_sum).reshape(-1, 1)
        inputs = torch.cat((input_sum - inputs, inputs), 1)

        return super().forward(inputs, labels, prior=prior)
