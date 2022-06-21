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
from sacrebleu.metrics import BLEU
from sacrebleu.utils import sum_of_lists
import re
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
from nltk.translate import meteor_score
from datasets.config import importlib_metadata, version


NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
if NLTK_VERSION >= version.Version("3.6.5"):
    from nltk import word_tokenize


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

    # def _download_and_prepare(self, dl_manager):
    #     import nltk
    #     nltk.download("wordnet")
    #     if NLTK_VERSION >= version.Version("3.6.5"):
    #         nltk.download("punkt")
    #     if NLTK_VERSION >= version.Version("3.6.6"):
    #         nltk.download("omw-1.4")

    def _compute(self, predictions, references):
        """Returns the scores: bleu"""
        metrics = {}
        # bleu
        bleu = BLEU(lowercase=True, force=False, tokenize=BLEU.TOKENIZER_DEFAULT, smooth_method='exp', smooth_value=None, effective_order=False)
        stats = sum_of_lists(bleu._extract_corpus_statistics(predictions, [references]))
        for n in range(1,5):
            metrics[f'bleu-{n}'] = bleu.compute_bleu(
                correct=stats[2: 2 + bleu.max_ngram_order],
                total=stats[2 + bleu.max_ngram_order:],
                sys_len=int(stats[0]), ref_len=int(stats[1]),
                smooth_method=bleu.smooth_method, smooth_value=bleu.smooth_value,
                effective_order=bleu.effective_order,
                max_ngram_order=n).score
                
        # unigram f1
        re_art = re.compile(r'\b(a|an|the)\b')
        re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
        stop_words = set(stopwords.words('english'))
        def utt2words(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            s = s.lower()
            s = re_punc.sub(' ', s)
            s = re_art.sub(' ', s)
            return s.split()

        metrics['unigram f1'] = []
        metrics['unigram f1 (non-stop words)'] = []
        for prediction, reference in zip(predictions, references):
            pred_items = utt2words(prediction)
            gold_items = utt2words(reference)
            for remove_stopwords in [False, True]:
                if remove_stopwords:
                    pred_items = [w for w in pred_items if w not in stop_words]
                    gold_items = [w for w in gold_items if w not in stop_words]
                common = Counter(pred_items) & Counter(gold_items)
                num_same = sum(common.values())
                if num_same == 0:
                    f1 = 0
                else:
                    precision = 1.0 * num_same / len(pred_items)
                    recall = 1.0 * num_same / len(gold_items)
                    f1 = (2 * precision * recall) / (precision + recall)
                if not remove_stopwords:
                    metrics['unigram f1'].append(f1)
                else:
                    metrics['unigram f1 (non-stop words)'].append(f1)
        metrics['unigram f1'] = np.mean(metrics['unigram f1'])
        metrics['unigram f1 (non-stop words)'] = np.mean(metrics['unigram f1 (non-stop words)'])

        # rouge-1/2/L-fmeasure
        rouge_types=["rouge1", "rouge2", "rougeL"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for prediction, reference in zip(predictions, references):
            score = scorer.score(reference, prediction)
            aggregator.add_scores(score)

        result = aggregator.aggregate()
        metrics.update({key: value.mid.fmeasure * 100 for key, value in result.items()})

        # meteor
        alpha=0.9
        beta=3
        gamma=0.5
        if NLTK_VERSION >= version.Version("3.6.5"):
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
                )
                for ref, pred in zip(references, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                for ref, pred in zip(references, predictions)
            ]
        metrics.update({"meteor": np.mean(scores)})

        # inter/intra-distinct-1/2
        def _ngram(seq, n):
            for i in range(len(seq) - n + 1):
                yield tuple(seq[i : i + n])
        
        for k in [1, 2]:
            inter_cnt = Counter()
            for prediction in predictions:
                ngram = Counter(_ngram(utt2words(prediction), k))
                inter_cnt += ngram
            metrics[f'distinct-{k}'] = max(len(inter_cnt), 1e-12) / max(sum(inter_cnt.values()), 1e-5)

        return metrics
