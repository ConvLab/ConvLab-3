# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""NLU Metric"""

import datasets
import re


# TODO: Add BibTeX citation
_CITATION = """\
"""

_DESCRIPTION = """\
Metric to evaluate text-to-text models on the natural language understanding task.
"""

_KWARGS_DESCRIPTION = """
Calculates sequence exact match, dialog acts accuracy and f1
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
Returns:
    seq_em: sequence exact match
    accuracy: dialog acts accuracy
    overall_f1: dialog acts overall f1
    binary_f1: binary dialog acts f1
    categorical_f1: categorical dialog acts f1
    non-categorical_f1: non-categorical dialog acts f1
Examples:

    >>> nlu_metric = datasets.load_metric("nlu_metric.py")
    >>> predictions = ["[binary]-[thank]-[general]-[]", "[non-categorical]-[inform]-[taxi]-[leave at]-[17:15]"]
    >>> references = ["[binary]-[thank]-[general]-[]", "[non-categorical]-[inform]-[train]-[leave at]-[17:15]"]
    >>> results = nlu_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'seq_em': 0.5, 'accuracy': 0.5, 
    'overall_f1': 0.5, 'overall_precision': 0.5, 'overall_recall': 0.5, 
    'binary_f1': 1.0, 'binary_precision': 1.0, 'binary_recall': 1.0, 
    'categorical_f1': 0.0, 'categorical_precision': 0.0, 'categorical_recall': 0.0, 
    'non-categorical_f1': 0.0, 'non-categorical_precision': 0.0, 'non-categorical_recall': 0.0}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NLUMetrics(datasets.Metric):
    """Metric to evaluate text-to-text models on the natural language understanding task."""

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            })
        )

    def deserialize_dialogue_acts(self, das_seq):
        dialogue_acts = {'binary': [], 'categorical': [], 'non-categorical': []}
        if len(das_seq) == 0:
            return dialogue_acts
        da_seqs = das_seq.split('];[')
        for i, da_seq in enumerate(da_seqs):
            if len(da_seq) == 0:
                continue
            if i == 0:
                if da_seq[0] == '[':
                    da_seq = da_seq[1:]
            if i == len(da_seqs) - 1:
                if da_seq[-1] == ']':
                    da_seq = da_seq[:-1]
            da = da_seq.split('][')
            if len(da) == 0:
                continue
            da_type = da[0]
            if len(da) == 5 and da_type in ['categorical', 'non-categorical']:
                dialogue_acts[da_type].append({'intent': da[1], 'domain': da[2], 'slot': da[3], 'value': da[4]})
            elif len(da) == 4 and da_type == 'binary':
                dialogue_acts[da_type].append({'intent': da[1], 'domain': da[2], 'slot': da[3]})
            else:
                # invalid da format, skip
                # print(das_seq)
                # print(da_seq)
                # print()
                pass
        return dialogue_acts

    def _compute(self, predictions, references):
        """Returns the scores: sequence exact match, dialog acts accuracy and f1"""
        seq_em = []
        acc = []
        f1_metrics = {x: {'TP':0, 'FP':0, 'FN':0} for x in ['overall', 'binary', 'categorical', 'non-categorical']}

        for prediction, reference in zip(predictions, references):
            seq_em.append(prediction.strip()==reference.strip())
            pred_da = self.deserialize_dialogue_acts(prediction)
            gold_da = self.deserialize_dialogue_acts(reference)
            flag = True
            for da_type in ['binary', 'categorical', 'non-categorical']:
                if da_type == 'binary':
                    predicts = [(x['intent'], x['domain'], x['slot']) for x in pred_da[da_type]]
                    labels = [(x['intent'], x['domain'], x['slot']) for x in gold_da[da_type]]
                else:
                    predicts = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in pred_da[da_type]]
                    labels = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in gold_da[da_type]]
                predicts = sorted(list(set(predicts)))
                labels = sorted(list(set(labels)))
                for ele in predicts:
                    if ele in labels:
                        f1_metrics['overall']['TP'] += 1
                        f1_metrics[da_type]['TP'] += 1
                    else:
                        f1_metrics['overall']['FP'] += 1
                        f1_metrics[da_type]['FP'] += 1
                for ele in labels:
                    if ele not in predicts:
                        f1_metrics['overall']['FN'] += 1
                        f1_metrics[da_type]['FN'] += 1
                flag &= (predicts==labels)
            acc.append(flag)

        for metric in list(f1_metrics.keys()):
            TP = f1_metrics[metric].pop('TP')
            FP = f1_metrics[metric].pop('FP')
            FN = f1_metrics[metric].pop('FN')
            precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
            recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
            f1_metrics.pop(metric)
            f1_metrics[f'{metric}_f1'] = f1
            f1_metrics[f'{metric}_precision'] = precision
            f1_metrics[f'{metric}_recall'] = recall

        return {
            "seq_em": sum(seq_em)/len(seq_em),
            "accuracy": sum(acc)/len(acc),
            **f1_metrics
        }
