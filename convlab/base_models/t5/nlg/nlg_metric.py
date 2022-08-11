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
"""NLG Metric"""

import datasets
import sacrebleu

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""

_DESCRIPTION = """\
Metric to evaluate text-to-text models on the natural language generation task.
"""

_KWARGS_DESCRIPTION = """
Calculates corpus-bleu4
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
Returns:
    bleu: corpus-bleu score
Examples:

    >>> nlg_metric = datasets.load_metric("nlg_metric.py")
    >>> predictions = ["hello there general kenobi", "foo bar foobar"]
    >>> references = ["hello there kenobi", "foo bar foobar"]
    >>> results = nlg_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'bleu': 35.35533905932737}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NLGMetrics(datasets.Metric):
    """Metric to evaluate text-to-text models on the natural language generation task."""

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
        """Returns the scores: bleu"""
        references = [" " if ref=="" else ref for ref in references]
        bleu = sacrebleu.corpus_bleu(predictions, [references], lowercase=True).score
        
        return {
            "bleu": bleu
        }
