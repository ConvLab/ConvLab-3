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
"""Metrics for DST models."""

import json
import os

import torch
from transformers.utils import ModelOutput
from matplotlib import pyplot as plt

from convlab.util import load_dataset, load_dst_data
from convlab.dst.setsumbt.datasets.utils import clean_states


class Metrics(ModelOutput):
    """Metrics for DST models."""
    def __add__(self, other):
        """Add two metrics objects."""
        for key, itm in other.items():
            assert key not in self
            self[key] = itm
        return self

    def compute_score(self, **weights):
        """
        Compute the score for the metrics object.

        Args:
            request (float): The weight for the request F1 score.
            active_domain (float): The weight for the active domain F1 score.
            general_act (float): The weight for the general act F1 score.
        """
        assert 'joint_goal_accuracy' in self
        self.score = 0.0
        if 'request_f1' in self and 'request' in weights:
            self.score += self.request_f1 * weights['request']
        if 'active_domain_f1' in self and 'active_domain' in weights:
            self.score += self.active_domain_f1 * weights['active_domain']
        if 'general_act_f1' in self and 'general_act' in weights:
            self.score += self.general_act_f1 * weights['general_act']
        self.score += self.joint_goal_accuracy

    def __gt__(self, other):
        """Compare two metrics objects."""
        assert isinstance(other, Metrics)

        if self.joint_goal_accuracy > other.joint_goal_accuracy:
            return True
        elif 'score' in self and 'score' in other and self.score > other.score:
            return True
        elif self.training_loss < other.training_loss:
            return True
        else:
            return False

class JointGoalAccuracy:
    """Joint goal accuracy metric."""

    def __init__(self, dataset_names, validation_split='test'):
        """
        Initialize the joint goal accuracy metric.

        Args:
            dataset_names (str): The name of the dataset(s) to use for computing the metric.
            validation_split (str): The split of the dataset to use for computing the metric.
        """
        self.dataset_names = [name for name in dataset_names.split('+')]
        self.validation_split = validation_split
        self._extract_data()
        self._extract_states()
        self._init_session()

    def _extract_data(self):
        """Extract the data from the dataset."""
        dataset_dicts = [load_dataset(dataset_name=name) for name in self.dataset_names]
        self.golden_states = dict()
        for dataset_dict in dataset_dicts:
            dataset = load_dst_data(dataset_dict, data_split=self.validation_split, speaker='all', dialogue_acts=True,
                                    split_to_turn=False)
            for dial in dataset[self.validation_split]:
                self.golden_states[dial['dialogue_id']] = dial['turns']

    @staticmethod
    def _clean_state(state):
        """
        Clean the state to remove pipe separated values and map values to the standard set.

        Args:
            state (dict): The state to clean.

        Returns:
            dict: The cleaned state.
        """

        turns = [{'dialogue_acts': list(),
                  'state': state}]
        turns = clean_states(turns)
        clean_state = turns[0]['state']
        clean_state = {domain: {slot: value if value != 'none' else '' for slot, value in domain_state.items()}
                       for domain, domain_state in clean_state.items()}

        return clean_state

    def _extract_states(self):
        """Extract the states from the dataset."""
        for dial_id, dial in self.golden_states.items():
            states = list()
            for turn in dial:
                if 'state' in turn:
                    state = self._clean_state(turn['state'])
                    states.append(state)
            self.golden_states[dial_id] = states

    def _init_session(self):
        """Initialize the session."""
        self.samples = dict()

    def add_dialogues(self, predictions):
        """
        Add dialogues to the metric.

        Args:
            predictions (dict): Dictionary of predicted dialogue belief states.
        """
        for dial_id, dialogue in predictions.items():
            for turn_id, turn in enumerate(dialogue):
                if dial_id in self.golden_states:
                    sample = {'dialogue_id': dial_id,
                              'turn_id': turn_id,
                              'state': self.golden_states[dial_id][turn_id],
                              'predictions': turn['belief_state']}
                    self.samples[f"{dial_id}_{turn_id}"] = sample

    def save_dialogues(self, path):
        """
        Save the dialogues and predictions to a file.

        Args:
            path (str): The path to save the dialogues to.
        """
        dialogues = list()
        for idx, turn in self.samples.items():
            predictions = dict()
            for domain in turn['state']:
                predictions[domain] = dict()
                for slot in turn['state'][domain]:
                    predictions[domain][slot] = turn['predictions'].get(domain, dict()).get(slot, '')
            dialogues.append({'dialogue_id': turn['dialogue_id'],
                              'turn_id': turn['turn_id'],
                              'state': turn['state'],
                              'predictions': {'state': predictions}})

        with open(path, 'w') as writer:
            json.dump(dialogues, writer, indent=2)
            writer.close()

    def evaluate(self):
        """Evaluate the metric."""
        assert len(self.samples) > 0
        metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'Correct': 0, 'N': 0}
        for dial_id, sample in self.samples.items():
            correct = True
            for domain in sample['state']:
                for slot, values in sample['state'][domain].items():
                    metrics['N'] += 1
                    if domain not in sample['predictions'] or slot not in sample['predictions'][domain]:
                        predict_values = ''
                    else:
                        predict_values = ''.join(sample['predictions'][domain][slot].split()).lower()
                    if len(values) > 0:
                        if len(predict_values) > 0:
                            values = [''.join(value.split()).lower() for value in values.split('|')]
                            predict_values = [''.join(value.split()).lower() for value in predict_values.split('|')]
                            if any([value in values for value in predict_values]):
                                metrics['TP'] += 1
                            else:
                                correct = False
                                metrics['FP'] += 1
                        else:
                            metrics['FN'] += 1
                            correct = False
                    else:
                        if len(predict_values) > 0:
                            metrics['FP'] += 1
                            correct = False
                        else:
                            metrics['TN'] += 1

            metrics['Correct'] += int(correct)

        TP = metrics.pop('TP')
        FP = metrics.pop('FP')
        FN = metrics.pop('FN')
        TN = metrics.pop('TN')
        Correct = metrics.pop('Correct')
        N = metrics.pop('N')
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        slot_accuracy = (TP + TN) / N
        joint_goal_accuracy = Correct / len(self.samples)

        metrics = Metrics(joint_goal_accuracy=joint_goal_accuracy * 100.,
                          slot_accuracy=slot_accuracy * 100.,
                          slot_f1=f1 * 100.,
                          slot_precision=precision * 100.,
                          slot_recall=recall * 100.)

        return metrics


class BeliefStateUncertainty:
    """Compute the uncertainty of the belief state predictions."""

    def __init__(self, n_confidence_bins=10):
        """
        Initialize the metric.

        Args:
            n_confidence_bins (int): Number of confidence bins.
        """
        self._init_session()
        self.n_confidence_bins = n_confidence_bins

    def _init_session(self):
        """Initialize the session."""
        self.samples = {'belief_state': dict(),
                        'golden_state': dict()}
        self.bin_info = {'confidence': None,
                         'accuracy': None}

    def add_dialogues(self, predictions, labels):
        """
        Add dialogues to the metric.

        Args:
            predictions (dict): Dictionary of predicted dialogue belief states.
            labels (dict): Dictionary of golden dialogue belief states.
        """
        for slot, probs in predictions.items():
            if slot not in self.samples['belief_state']:
                self.samples['belief_state'][slot] = probs.reshape(-1, probs.size(-1)).cpu()
                self.samples['golden_state'][slot] = labels[slot].reshape(-1).cpu()
            else:
                self.samples['belief_state'][slot] = torch.cat((self.samples['belief_state'][slot],
                                                                probs.reshape(-1, probs.size(-1)).cpu()), 0)
                self.samples['golden_state'][slot] = torch.cat((self.samples['golden_state'][slot],
                                                                labels[slot].reshape(-1).cpu()), 0)

    def _fill_bins(self, probs: torch.Tensor) -> list:
        """
        Fill the bins with the relevant observation ids.

        Args:
            probs (Tensor): Predicted probabilities.

        Returns:
            list: List of bins.
        """
        assert probs.dim() == 2
        probs = probs.max(-1)[0]

        step = 1.0 / self.n_confidence_bins
        bin_ranges = torch.arange(0.0, 1.0 + 1e-10, step)
        bins = []
        # Compute the bin ranges
        for b in range(self.n_confidence_bins):
            lower, upper = bin_ranges[b], bin_ranges[b + 1]
            if b == 0:
                ids = torch.where((probs >= lower) * (probs <= upper))[0]
            else:
                ids = torch.where((probs > lower) * (probs <= upper))[0]
            bins.append(ids)

        return bins

    @staticmethod
    def _bin_confidence(bins: list, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the average confidence score for each bin.

        Args:
            bins (list): List of confidence bins.
            probs (Tensor): Predicted probabilities.

        Returns:
            scores: Confidence score for each bin.
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

    def _jg_ece(self) -> float:
        """Compute the joint goal Expected Calibration Error."""
        y_pred = {slot: probs.argmax(-1) for slot, probs in self.samples['belief_state'].items()}
        goal_acc = [(y_pred[slot] == y_true).int() for slot, y_true in self.samples['golden_state'].items()]
        goal_acc = (sum(goal_acc) / len(goal_acc)).int()

        # Confidence score is minimum across slots as a single bad predictions leads to incorrect prediction in state
        scores = [probs.max(-1)[0].unsqueeze(1) for slot, probs in self.samples["belief_state"].items()]
        scores = torch.cat(scores, 1).min(1)[0]

        bins = self._fill_bins(scores.unsqueeze(-1))
        conf = self._bin_confidence(bins, scores.unsqueeze(-1))

        slot_0 = list(self.samples['golden_state'].keys())[0]
        acc = []
        for b in bins:
            if b is not None:
                acc_ = goal_acc[b]
                acc_ = acc_[self.samples['golden_state'][slot_0][b] >= 0]
                if acc_.size(0) >= 0:
                    acc.append(acc_.float().mean())
                else:
                    acc.append(-1)
            else:
                acc.append(-1)
        acc = torch.tensor(acc)

        self.bin_info['confidence'] = conf
        self.bin_info['accuracy'] = acc

        n = self.samples["belief_state"][slot_0].size(0)
        bk = torch.tensor([b.size(0) for b in bins])

        ece = torch.abs(conf - acc) * bk / n
        ece = ece[acc >= 0.0]
        ece = ece.sum().item()

        return ece

    def draw_calibration_diagram(self, save_path: str, validation_split=None):
        """
        Draw the calibration diagram.

        Args:
            save_path (str): Path to save the calibration diagram.
            validation_split (str): Validation split.
        """
        if self.bin_info['confidence'] is None:
            self._jg_ece()

        acc = self.bin_info['accuracy']
        conf = self.bin_info['confidence']
        conf = conf[acc >= 0.0]
        acc = acc[acc >= 0.0]

        fig = plt.figure(figsize=(14,8))
        font = 20
        plt.tick_params(labelsize=font - 2)
        linestyle = '-'

        plt.plot(torch.tensor([0, 1]), torch.tensor([0, 1]), linestyle='--', color='black', linewidth=3)
        plt.plot(conf, acc, linestyle=linestyle, color='red', linewidth=3)
        plt.xlabel('Confidence', fontsize=font)
        plt.ylabel('Joint Goal Accuracy', fontsize=font)

        path = validation_split + '_calibration_diagram.json' if validation_split else 'calibration_diagram.json'
        path = os.path.join(save_path, 'predictions', path)
        with open(path, 'w') as f:
            json.dump({'confidence': conf.tolist(), 'accuracy': acc.tolist()}, f)

        path = validation_split + '_calibration_diagram.png' if validation_split else 'calibration_diagram.png'
        path = os.path.join(save_path, path)
        plt.savefig(path)

    def _l2_err(self, remove_belief: bool = False) -> float:
        """
        Compute the L2 error between the predicted and target distribution.

        Args:
            remove_belief (bool): Remove the belief state and replace it with a 1 hot prediction.

        Returns:
            l2_err: L2 error between the predicted and target distribution.
        """
        # Get ids used for removing padding turns.
        slot_0 = list(self.samples['golden_state'].keys())[0]
        padding = torch.where(self.samples['golden_state'][slot_0] != -1)[0]

        distributions = []
        labels = []
        for slot, probs in self.samples['belief_state'].items():
            # Replace distribution by a 1 hot prediction
            if remove_belief:
                probs_ = torch.zeros(probs.shape).float()
                probs_[range(probs.size(0)), probs.argmax(-1)] = 1.0
                probs = probs_
                del probs_
            # Remove padding turns
            lab = self.samples['golden_state'][slot]
            probs = probs[padding]
            lab = lab[padding]

            # Target distribution
            y = torch.zeros(probs.shape)
            y[range(y.size(0)), lab] = 1.0

            distributions.append(probs)
            labels.append(y)

        # Concatenate all slots into a single belief state
        distributions = torch.cat(distributions, -1)
        labels = torch.cat(labels, -1)

        # Calculate L2-Error for each turn
        err = torch.sqrt(((labels - distributions) ** 2).sum(-1))
        return err.mean().item()

    def evaluate(self):
        """Evaluate the metrics."""
        l2_err = self._l2_err(remove_belief=False)
        binary_l2_err = self._l2_err(remove_belief=True)
        l2_err_ratio = (binary_l2_err - l2_err) / binary_l2_err
        metrics = Metrics(
            joint_goal_ece=self._jg_ece() * 100.,
            joint_l2_error=l2_err,
            joint_l2_error_ratio=l2_err_ratio * 100.
        )
        return metrics


class ActPredictionAccuracy:
    """Calculate the accuracy of the action predictions."""

    def __init__(self, act_type, binary=False):
        """
        Args:
            act_type (str): Type of action to evaluate.
            binary (bool): Whether the action is binary or multilabel.
        """
        self.act_type = act_type
        self.binary = binary
        self._init_session()

    def _init_session(self):
        """Initialize the session."""
        self.samples = {'predictions': dict(),
                        'labels': dict()}

    def add_dialogues(self, predictions, labels):
        """
        Add dialogues to the session.

        Args:
            predictions (dict): Action predictions.
            labels (dict): Action labels.
        """
        for slot, probs in predictions.items():
            if slot in labels:
                pred = probs.cpu().argmax(-1).reshape(-1) if not self.binary else probs.cpu().round().int().reshape(-1)
                if slot not in self.samples['predictions']:
                    self.samples['predictions'][slot] = pred
                    self.samples['labels'][slot] = labels[slot].reshape(-1).cpu()
                else:
                    self.samples['predictions'][slot] = torch.cat((self.samples['predictions'][slot], pred), 0)
                    self.samples['labels'][slot] = torch.cat((self.samples['labels'][slot],
                                                              labels[slot].reshape(-1).cpu()), 0)

    def evaluate(self):
        """Evaluate the metrics."""
        metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'Correct': 0, 'N': 0}
        for slot, pred in self.samples['predictions'].items():
            metrics['N'] += pred.size(0)
            metrics['Correct'] += (pred == self.samples['labels'][slot]).sum()
            tp = (pred > 0) * (self.samples['labels'][slot] > 0) * (pred == self.samples['labels'][slot])
            metrics['TP'] += tp.sum()
            metrics['FP'] += ((pred > 0) * (self.samples['labels'][slot] == 0)).sum()
            metrics['FN'] += ((pred == 0) * (self.samples['labels'][slot] > 0)).sum()

        TP = metrics.pop('TP')
        FP = metrics.pop('FP')
        FN = metrics.pop('FN')
        Correct = metrics.pop('Correct')
        N = metrics.pop('N')
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.

        metrics = {f'{self.act_type}_f1': f1 * 100.}
        return Metrics(**metrics)
