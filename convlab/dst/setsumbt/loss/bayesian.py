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
"""Bayesian Matching Activation and Loss Functions"""

import torch
from torch import digamma, lgamma
from torch.nn import Module


# Inverse Linear activation function
def invlinear(x):
    z = (1.0 / (1.0 - x)) * (x < 0)
    z += (1.0 + x) * (x >= 0)
    return z

# Exponential activation function
def exponential(x):
    return torch.exp(x)


# Dirichlet activation function for the model
def dirichlet(a):
    p = exponential(a)
    repeat_dim = (1,)*(len(p.shape)-1) + (p.size(-1),)
    p = p / p.sum(-1).unsqueeze(-1).repeat(repeat_dim)
    return p


# Pytorch BayesianMatchingLoss nn.Module
class BayesianMatchingLoss(Module):

    def __init__(self, lamb=0.01, ignore_index=-1):
        super(BayesianMatchingLoss, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, alpha, labels, prior=None):
        # Assert input sizes
        assert alpha.dim() == 2                 # Observations, predictive distribution
        assert labels.dim() == 1                # Label for each observation
        assert labels.size(0) == alpha.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if labels.max() <= alpha.size(-1):
            dimension = alpha.size(-1)
        else:
            raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.max(), alpha.size(-1)))
        
        # Remove observations with no labels
        if prior is not None:
            prior = prior[labels != self.ignore_index]
        alpha = exponential(alpha[labels != self.ignore_index])
        labels = labels[labels != self.ignore_index]
        
        # Initialise and reshape prior parameters
        if prior is None:
            prior = torch.ones(dimension)
        prior = prior.to(alpha.device)

        # KL divergence term
        lb = lgamma(alpha.sum(-1)) - lgamma(prior.sum(-1)) + (lgamma(prior) - lgamma(alpha)).sum(-1)
        e = digamma(alpha) - digamma(alpha.sum(-1)).unsqueeze(-1).repeat((1, alpha.size(-1)))
        e = ((alpha - prior) * e).sum(-1)
        kl = lb + e
        kl *= self.lamb
        del lb, e, prior

        # Expected log likelihood
        expected_likelihood = digamma(alpha[range(labels.size(0)), labels]) - digamma(alpha.sum(1))
        del alpha, labels

        # Apply ELBO loss and mean reduction
        loss = (kl - expected_likelihood).mean()
        del kl, expected_likelihood

        return loss


# Pytorch BayesianMatchingLoss nn.Module
class BinaryBayesianMatchingLoss(Module):

    def __init__(self, lamb=0.01, ignore_index=-1):
        super(BinaryBayesianMatchingLoss, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, alpha, labels, prior=None):
        # Assert input sizes
        assert alpha.dim() == 1                 # Observations, predictive distribution
        assert labels.dim() == 1                # Label for each observation
        assert labels.size(0) == alpha.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if labels.max() <= 2:
            dimension = 2
        else:
            raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.max(), alpha.size(-1)))
        
        # Remove observations with no labels
        if prior is not None:
            prior = prior[labels != self.ignore_index]
        alpha = alpha[labels != self.ignore_index]
        alpha_sum = 1 + (1 / self.lamb)
        alpha = (torch.sigmoid(alpha) * alpha_sum).reshape(-1, 1)
        alpha = torch.cat((alpha_sum - alpha, alpha), 1)
        labels = labels[labels != self.ignore_index]
        
        # Initialise and reshape prior parameters
        if prior is None:
            prior = torch.ones(dimension)
        prior = prior.to(alpha.device)

        # KL divergence term
        lb = lgamma(alpha.sum(-1)) - lgamma(prior.sum(-1)) + (lgamma(prior) - lgamma(alpha)).sum(-1)
        e = digamma(alpha) - digamma(alpha.sum(-1)).unsqueeze(-1).repeat((1, alpha.size(-1)))
        e = ((alpha - prior) * e).sum(-1)
        kl = lb + e
        kl *= self.lamb
        del lb, e, prior

        # Expected log likelihood
        expected_likelihood = digamma(alpha[range(labels.size(0)), labels.long()]) - digamma(alpha.sum(1))
        del alpha, labels

        # Apply ELBO loss and mean reduction
        loss = (kl - expected_likelihood).mean()
        del kl, expected_likelihood

        return loss
