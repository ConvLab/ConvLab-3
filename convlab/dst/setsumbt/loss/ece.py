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
"""Expected calibration error"""

import torch


def fill_bins(n_bins, logits):
    assert logits.dim() == 2
    logits = logits.max(-1)[0]

    step = 1.0 / n_bins
    bin_ranges = torch.arange(0.0, 1.0 + 1e-10, step)
    bins = []
    for b in range(n_bins):
        lower, upper = bin_ranges[b], bin_ranges[b + 1]
        if b == 0:
            ids = torch.where((logits >= lower) * (logits <= upper))[0]
        else:
            ids = torch.where((logits > lower) * (logits <= upper))[0]
        bins.append(ids)
    return bins


def bin_confidence(bins, logits):
    logits = logits.max(-1)[0]

    scores = []
    for b in bins:
        if b is not None:
            l = logits[b]
            scores.append(l.mean())
        else:
            scores.append(-1)
    scores = torch.tensor(scores)
    return scores


def bin_accuracy(bins, logits, y_true):
    y_pred = logits.argmax(-1)

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


def ece(logits, y_true, n_bins):
    bins = fill_bins(n_bins, logits)

    scores = bin_confidence(bins, logits)
    acc = bin_accuracy(bins, logits, y_true)

    n = logits.size(0)
    bk = torch.tensor([b.size(0) for b in bins])

    ece = torch.abs(scores - acc) * bk / n
    ece = ece[acc >= 0.0]
    ece = ece.sum().item()

    return ece


def jg_ece(logits, y_true, n_bins):
    y_pred = {slot: logits[slot].reshape(-1, logits[slot].size(-1)).argmax(-1) for slot in logits}
    goal_acc = {slot: (y_pred[slot] == y_true[slot].reshape(-1)).int() for slot in y_pred}
    goal_acc = sum([goal_acc[slot] for slot in goal_acc])
    goal_acc = (goal_acc == len(y_true)).int()

    scores = [logits[slot].reshape(-1, logits[slot].size(-1)).max(-1)[0].unsqueeze(0) for slot in logits]
    scores = torch.cat(scores, 0).min(0)[0]

    step = 1.0 / n_bins
    bin_ranges = torch.arange(0.0, 1.0 + 1e-10, step)
    bins = []
    for b in range(n_bins):
        lower, upper = bin_ranges[b], bin_ranges[b + 1]
        if b == 0:
            ids = torch.where((scores >= lower) * (scores <= upper))[0]
        else:
            ids = torch.where((scores > lower) * (scores <= upper))[0]
        bins.append(ids)

    conf = []
    for b in bins:
        if b is not None:
            l = scores[b]
            conf.append(l.mean())
        else:
            conf.append(-1)
    conf = torch.tensor(conf)

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

    n = logits[slot].reshape(-1, logits[slot].size(-1)).size(0)
    bk = torch.tensor([b.size(0) for b in bins])

    ece = torch.abs(conf - acc) * bk / n
    ece = ece[acc >= 0.0]
    ece = ece.sum().item()

    return ece


def l2_acc(belief_state, labels, remove_belief=False):
    # Get ids used for removing padding turns.
    padding = labels[list(labels.keys())[0]].reshape(-1)
    padding = torch.where(padding != -1)[0]

    # l2 = []
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

        # err = torch.sqrt(((y - bs) ** 2).sum(-1))
        # l2.append(err.unsqueeze(-1))

        state.append(bs)
        labs.append(y)
    
    # err = torch.cat(l2, -1).max(-1)[0]

    # Concatenate all slots into a single belief state
    state = torch.cat(state, -1)
    labs = torch.cat(labs, -1)

    # Calculate L2-Error for each turn
    err = torch.sqrt(((labs - state) ** 2).sum(-1))
    return err.mean()
