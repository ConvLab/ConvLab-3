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
from torch import lgamma, log
from torch.nn import Module
from torch.nn.functional import kl_div

from convlab2.dst.setsumbt.loss.bayesian import BayesianMatchingLoss


# Pytorch BayesianMatchingLoss nn.Module
class DistillationKL(Module):

    def __init__(self, lamb=1e-4, ignore_index=-1):
        super(DistillationKL, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, alpha, labels, temp=1.0):
        # Assert input sizes
        assert alpha.dim() == 2                 # Observations, predictive distribution
        assert labels.dim() == 2                # Label for each observation
        assert labels.size(0) == alpha.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        if labels.size(-1) == alpha.size(-1):
            dimension = alpha.size(-1)
        else:
            raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.size(-1), alpha.size(-1)))
        
        alpha = torch.log(torch.softmax(alpha / temp, -1))
        ids = torch.where(labels[:, 0] != self.ignore_index)[0]
        alpha = alpha[ids]
        labels = labels[ids]

        labels = ((1 - self.lamb) * labels) + (self.lamb * (1 / labels.size(-1)))

        kl = kl_div(alpha, labels, reduction='none').sum(-1).mean()
        return kl    


# Pytorch BayesianMatchingLoss nn.Module
class BinaryDistillationKL(Module):

    def __init__(self, lamb=1e-4, ignore_index=-1):
        super(BinaryDistillationKL, self).__init__()

        self.lamb = lamb
        self.ignore_index = ignore_index
    
    def forward(self, alpha, labels, temp=0.0):
        # Assert input sizes
        assert alpha.dim() == 1                 # Observations, predictive distribution
        assert labels.dim() == 1                # Label for each observation
        assert labels.size(0) == alpha.size(0)  # Equal number of observation

        # Confirm predictive distribution dimension
        # if labels.size(-1) == alpha.size(-1):
        #     dimension = alpha.size(-1)
        # else:
        #     raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.size(-1), alpha.size(-1)))
        
        alpha = torch.sigmoid(alpha / temp).unsqueeze(-1)
        ids = torch.where(labels != self.ignore_index)[0]
        alpha = alpha[ids]
        labels = labels[ids]

        alpha = torch.log(torch.cat((1 - alpha, alpha), 1))
        
        labels = labels.unsqueeze(-1)
        labels = torch.cat((1 - labels, labels), -1)
        labels = ((1 - self.lamb) * labels) + (self.lamb * (1 / labels.size(-1)))

        kl = kl_div(alpha, labels, reduction='none').sum(-1).mean()
        return kl  


# def smart_sort(x, permutation):
#     assert x.dim() == permutation.dim()
#     if x.dim() == 3:
#         d1, d2, d3 = x.size()
#         ret = x[torch.arange(d1).unsqueeze(-1).unsqueeze(-1).repeat((1, d2, d3)).flatten(),
#                 torch.arange(d2).unsqueeze(0).unsqueeze(-1).repeat((d1, 1, d3)).flatten(),
#                 permutation.flatten()].view(d1, d2, d3)
#         return ret
#     elif x.dim() == 2:
#         d1, d2 = x.size()
#         ret = x[torch.arange(d1).unsqueeze(-1).repeat((1, d2)).flatten(),
#                 permutation.flatten()].view(d1, d2)
#         return ret


# # Pytorch BayesianMatchingLoss nn.Module
# class DistillationNLL(Module):

#     def __init__(self, lamb=1e-4, ignore_index=-1):
#         super(DistillationNLL, self).__init__()

#         self.lamb = lamb
#         self.ignore_index = ignore_index
#         self.loss_add = BayesianMatchingLoss(lamb=0.001)
    
#     def forward(self, alpha, labels, temp=1.0):
#         # Assert input sizes
#         assert alpha.dim() == 2                 # Observations, predictive distribution
#         assert labels.dim() == 3                # Label for each observation
#         assert labels.size(0) == alpha.size(0)  # Equal number of observation

#         # Confirm predictive distribution dimension
#         if labels.size(-1) == alpha.size(-1):
#             dimension = alpha.size(-1)
#         else:
#             raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.size(-1), alpha.size(-1)))
        
#         alpha = torch.exp(alpha / temp)
#         ids = torch.where(labels[:, 0, 0] != self.ignore_index)[0]
#         alpha = alpha[ids]
#         labels = labels[ids]

#         best_labels = labels.mean(-2).argmax(-1)
#         loss2 = self.loss_add(alpha, best_labels)

#         topn = labels.mean(-2).argsort(-1, descending=True)
#         n = 10
#         alpha = smart_sort(alpha, topn)[:, :n]
#         labels = smart_sort(labels, topn.unsqueeze(-2).repeat((1, labels.size(-2), 1)))
#         labels = labels[:, :, :n]
#         labels = labels / labels.sum(-1).unsqueeze(-1).repeat((1, 1, labels.size(-1)))

#         labels = log(((1 - self.lamb) * labels) + (self.lamb * (1 / labels.size(-1))))

#         loss = (alpha - 1) * labels.mean(-2)
#         # loss = (alpha - 1) * labels
#         loss = lgamma(alpha.sum(-1)) - lgamma(alpha).sum(-1) + loss.sum(-1) 
#         loss = -1.0 * loss.mean()
#         # loss = -1.0 * loss.mean() / alpha.size(-1)

#         return loss      


# # Pytorch BayesianMatchingLoss nn.Module
# class BinaryDistillationNLL(Module):

#     def __init__(self, lamb=1e-4, ignore_index=-1):
#         super(BinaryDistillationNLL, self).__init__()

#         self.lamb = lamb
#         self.ignore_index = ignore_index
    
#     def forward(self, alpha, labels, temp=0.0):
#         # Assert input sizes
#         assert alpha.dim() == 1                 # Observations, predictive distribution
#         assert labels.dim() == 2                # Label for each observation
#         assert labels.size(0) == alpha.size(0)  # Equal number of observation

#         # Confirm predictive distribution dimension
#         # if labels.size(-1) == alpha.size(-1):
#         #     dimension = alpha.size(-1)
#         # else:
#         #     raise NameError('Label dimension %i is larger than prediction dimension %i.' % (labels.size(-1), alpha.size(-1)))
        
#         # Remove observations with no labels
#         ids = torch.where(labels[:, 0] != self.ignore_index)[0]
#         # alpha_sum = 1 + (1 / self.lamb)
#         alpha_sum = 10.0
#         alpha = (torch.sigmoid(alpha) * alpha_sum).reshape(-1, 1)
#         alpha = alpha[ids]
#         labels = labels[ids]

#         if temp != 1.0:
#             alpha = torch.log(alpha + 1e-4)
#             alpha = torch.exp(alpha / temp)

#         alpha = torch.cat((alpha_sum - alpha, alpha), 1)
        
#         labels = labels.unsqueeze(-1)
#         labels = torch.cat((1 - labels, labels), -1)
#         # labels[labels[:, 0, 0] == self.ignore_index] = 1
#         labels = log(((1 - self.lamb) * labels) + (self.lamb * (1 / labels.size(-1))))

#         loss = (alpha - 1) * labels.mean(-2)
#         loss = lgamma(alpha.sum(-1)) - lgamma(alpha).sum(-1) + loss.sum(-1)
#         loss = -1.0 * loss.mean()

#         return loss    
