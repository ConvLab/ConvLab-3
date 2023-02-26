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
"""Uncertainty evaluation metrics for dialogue belief tracking"""

import torch


def fill_bins(n_bins: int, probs: torch.Tensor) -> list:
    """
    Function to split observations into bins based on predictive probabilities

    Args:
        n_bins (int): Number of bins
        probs (Tensor): Predictive probabilities for the observations

    Returns:
        bins (list): List of observation ids for each bin
    """
    assert probs.dim() == 2
    probs = probs.max(-1)[0]

    step = 1.0 / n_bins
    bin_ranges = torch.arange(0.0, 1.0 + 1e-10, step)
    bins = []
    for b in range(n_bins):
        lower, upper = bin_ranges[b], bin_ranges[b + 1]
        if b == 0:
            ids = torch.where((probs >= lower) * (probs <= upper))[0]
        else:
            ids = torch.where((probs > lower) * (probs <= upper))[0]
        bins.append(ids)
    return bins


def bin_confidence(bins: list, probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the confidence score within each bin

    Args:
        bins (list): List of observation ids for each bin
        probs (Tensor): Predictive probabilities for the observations

    Returns:
        scores (Tensor): Average confidence score within each bin
    """
    probs = probs.max(-1)[0]

    scores = []
    for b in bins:
        if b is not None:
            scores.append(probs[b].mean())
        else:
            scores.append(-1)
    scores = torch.tensor(scores)
    return scores


def bin_accuracy(bins: list, probs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy score for observations in each bin

    Args:
        bins (list): List of observation ids for each bin
        probs (Tensor): Predictive probabilities for the observations
        y_true (Tensor): Labels for the observations

    Returns:
        acc (Tensor): Accuracies for the observations in each bin
    """
    y_pred = probs.argmax(-1)

    acc = []
    for b in bins:
        if b is not None:
            p = y_pred[b]
            acc_ = (p == y_true[b]).float()
            acc_ = acc_[y_true[b] >= 0]
            if acc_.size(0) >= 0:
                acc.append(acc_.mean())
            else:
                acc.append(-1)
        else:
            acc.append(-1)
    acc = torch.tensor(acc)
    return acc


def ece(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int) -> float:
    """
    Expected calibration error calculation

    Args:
        probs (Tensor): Predictive probabilities for the observations
        y_true (Tensor): Labels for the observations
        n_bins (int): Number of bins

    Returns:
        ece (float): Expected calibration error
    """
    bins = fill_bins(n_bins, probs)

    scores = bin_confidence(bins, probs)
    acc = bin_accuracy(bins, probs, y_true)

    n = probs.size(0)
    bk = torch.tensor([b.size(0) for b in bins])

    ece = torch.abs(scores - acc) * bk / n
    ece = ece[acc >= 0.0]
    ece = ece.sum().item()

    return ece


def jg_ece(belief_state: dict, y_true: dict, n_bins: int) -> float:
    """
        Joint goal expected calibration error calculation

        Args:
            belief_state (dict): Belief state probabilities for the dialogue turns
            y_true (dict): Labels for the state in dialogue turns
            n_bins (int): Number of bins

        Returns:
            ece (float): Joint goal expected calibration error
        """
    y_pred = {slot: bs.reshape(-1, bs.size(-1)).argmax(-1) for slot, bs in belief_state.items()}
    goal_acc = {slot: (y_pred[slot] == y_true[slot].reshape(-1)).int() for slot in y_pred}
    goal_acc = sum([goal_acc[slot] for slot in goal_acc])
    goal_acc = (goal_acc == len(y_true)).int()

    # Confidence score is minimum across slots as a single bad predictions leads to incorrect prediction in state
    scores = [bs.reshape(-1, bs.size(-1)).max(-1)[0].unsqueeze(0) for slot, bs in belief_state.items()]
    scores = torch.cat(scores, 0).min(0)[0]

    bins = fill_bins(n_bins, scores.unsqueeze(-1))

    conf = bin_confidence(bins, scores.unsqueeze(-1))

    slot = [s for s in y_true][0]
    acc = []
    for b in bins:
        if b is not None:
            acc_ = goal_acc[b]
            acc_ = acc_[y_true[slot].reshape(-1)[b] >= 0]
            if acc_.size(0) >= 0:
                acc.append(acc_.float().mean())
            else:
                acc.append(-1)
        else:
            acc.append(-1)
    acc = torch.tensor(acc)

    n = belief_state[slot].reshape(-1, belief_state[slot].size(-1)).size(0)
    bk = torch.tensor([b.size(0) for b in bins])

    ece = torch.abs(conf - acc) * bk / n
    ece = ece[acc >= 0.0]
    ece = ece.sum().item()

    return ece


def l2_acc(belief_state: dict, labels: dict, remove_belief: bool = False) -> float:
    """
    Compute L2 Error of belief state prediction

    Args:
        belief_state (dict): Belief state probabilities for the dialogue turns
        labels (dict): Labels for the state in dialogue turns
        remove_belief (bool): Convert belief state to dialogue state

    Returns:
        err (float): L2 Error of belief state prediction
    """
    # Get ids used for removing padding turns.
    padding = labels[list(labels.keys())[0]].reshape(-1)
    padding = torch.where(padding != -1)[0]

    state = []
    labs = []
    for slot, bs in belief_state.items():
        # Predictive Distribution
        bs = bs.reshape(-1, bs.size(-1)).cuda()
        # Replace distribution by a 1 hot prediction
        if remove_belief:
            bs_ = torch.zeros(bs.shape).float().cuda()
            bs_[range(bs.size(0)), bs.argmax(-1)] = 1.0
            bs = bs_
            del bs_
        # Remove padding turns
        lab = labels[slot].reshape(-1).cuda()
        bs = bs[padding]
        lab = lab[padding]

        # Target distribution
        y = torch.zeros(bs.shape).cuda()
        y[range(y.size(0)), lab] = 1.0

        state.append(bs)
        labs.append(y)

    # Concatenate all slots into a single belief state
    state = torch.cat(state, -1)
    labs = torch.cat(labs, -1)

    # Calculate L2-Error for each turn
    err = torch.sqrt(((labs - state) ** 2).sum(-1))
    return err.mean()
