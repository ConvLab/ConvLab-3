# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Calibration Plot plotting script"""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='Location of the belief states', required=True)
    parser.add_argument('--output', help='Output image path', default='calibration_plot.png')
    parser.add_argument('--n_bins', help='Number of bins', default=10, type=int)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    path = args.data_dir

    models = os.listdir(path)
    models = [os.path.join(path, model, 'test.predictions') for model in models]

    fig = plt.figure(figsize=(14,8))
    font=20
    plt.tick_params(labelsize=font-2)
    linestyle = ['-', ':', (0, (3, 5, 1, 5)), '-.', (0, (5, 10))]
    for i, model in enumerate(models):
        conf, acc = get_calibration(model, device, n_bins=args.n_bins)
        name = model.split('/')[-2].strip()
        print(name, conf, acc)
        plt.plot(conf, acc, label=name, linestyle=linestyle[i], linewidth=3)

    plt.plot(torch.tensor([0,1]), torch.tensor([0,1]), linestyle='--', color='black', linewidth=3)
    plt.xlabel('Confidence', fontsize=font)
    plt.ylabel('Joint Goal Accuracy', fontsize=font)
    plt.legend(fontsize=font)

    plt.savefig(args.output)


def get_calibration(path, device, n_bins=10, temperature=1.00):
    probs = torch.load(path, map_location=device)
    y_true = probs['state_labels']
    probs = probs['belief_states']

    y_pred = {slot: probs[slot].reshape(-1, probs[slot].size(-1)).argmax(-1) for slot in probs}
    goal_acc = {slot: (y_pred[slot] == y_true[slot].reshape(-1)).int() for slot in y_pred}
    goal_acc = sum([goal_acc[slot] for slot in goal_acc])
    goal_acc = (goal_acc == len(y_true)).int()

    scores = [probs[slot].reshape(-1, probs[slot].size(-1)).max(-1)[0].unsqueeze(0) for slot in probs]
    scores = torch.cat(scores, 0).min(0)[0]

    step = 1.0 / float(n_bins)
    bin_ranges = torch.arange(0.0, 1.0 + 1e-10, step)
    bins = []
    for b in range(n_bins):
        lower, upper = bin_ranges[b], bin_ranges[b + 1]
        if b == 0:
            ids = torch.where((scores >= lower) * (scores <= upper))[0]
        else:
            ids = torch.where((scores > lower) * (scores <= upper))[0]
        bins.append(ids)

    conf = [0.0]
    for b in bins:
        if b.size(0) > 0:
            l = scores[b]
            conf.append(l.mean())
        else:
            conf.append(-1)
    conf = torch.tensor(conf)

    slot = [s for s in y_true][0]
    acc = [0.0]
    for b in bins:
        if b.size(0) > 0:
            acc_ = goal_acc[b]
            acc_ = acc_[y_true[slot].reshape(-1)[b] >= 0]
            if acc_.size(0) >= 0:
                acc.append(acc_.float().mean())
            else:
                acc.append(-1)
        else:
            acc.append(-1)
    acc = torch.tensor(acc)

    conf = conf[acc != -1]
    acc = acc[acc != -1]

    return conf, acc


if __name__ == '__main__':
    main()
