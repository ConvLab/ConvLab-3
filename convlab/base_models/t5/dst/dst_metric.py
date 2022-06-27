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
"""DST Metric"""

import datasets
from convlab.base_models.t5.dst.serialization import deserialize_dialogue_state


# TODO: Add BibTeX citation
_CITATION = """\
"""

_DESCRIPTION = """\
Metric to evaluate text-to-text models on the dialog state tracking task.
"""

_KWARGS_DESCRIPTION = """
Calculates sequence exact match, joint goal accuracy and slot f1
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
Returns:
    seq_em: sequence exact match
    accuracy: dialog state accuracy
    slot_f1: slot f1
Examples:

    >>> dst_metric = datasets.load_metric("dst_metric.py")
    >>> predictions = ["[restaurant][price range][moderate]", "[restaurant][price range][moderate];[restaurant][food][catalan];[restaurant][area][centre]"]
    >>> references = ["[restaurant][price range][moderate]", "[restaurant][price range][moderate];[restaurant][food][catalan];[attraction][area][centre]"]
    >>> results = dst_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'seq_em': 0.5, 'accuracy': 0.5, 
    'slot_f1': 0.75, 'slot_precision': 0.75, 'slot_recall': 0.75}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DSTMetrics(datasets.Metric):
    """Metric to evaluate text-to-text models on the dialog state tracking task."""

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

    def _compute(self, predictions, references):
        """Returns the scores: sequence exact match, joint goal accuracy and slot f1"""
        seq_em = []
        acc = []
        f1_metrics = {'TP':0, 'FP':0, 'FN':0}

        for prediction, reference in zip(predictions, references):
            seq_em.append(prediction.strip()==reference.strip())
            pred_state = deserialize_dialogue_state(prediction)
            gold_state = deserialize_dialogue_state(reference)

            predicts = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in pred_state for slot, value in pred_state[domain].items() if len(value)>0}))
            labels = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in gold_state for slot, value in gold_state[domain].items() if len(value)>0}))

            flag = True
            for ele in predicts:
                if ele in labels:
                    f1_metrics['TP'] += 1
                else:
                    f1_metrics['FP'] += 1
            for ele in labels:
                if ele not in predicts:
                    f1_metrics['FN'] += 1
            flag &= (predicts==labels)
            acc.append(flag)

        TP = f1_metrics.pop('TP')
        FP = f1_metrics.pop('FP')
        FN = f1_metrics.pop('FN')
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        f1_metrics[f'slot_f1'] = f1
        f1_metrics[f'slot_precision'] = precision
        f1_metrics[f'slot_recall'] = recall

        return {
            "seq_em": sum(seq_em)/len(seq_em),
            "accuracy": sum(acc)/len(acc),
            **f1_metrics
        }
