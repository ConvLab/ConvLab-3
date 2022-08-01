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
"""Grounded Dialog Generation Metric"""

from weakref import ref
import datasets
from sacrebleu.metrics import BLEU
from sacrebleu.utils import sum_of_lists
import re
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from rouge_score import rouge_scorer, scoring
from nltk.translate import meteor_score
from datasets.config import importlib_metadata, version
from convlab.base_models.t5.key2gen.features import FEATURES
from convlab.util import load_ontology
from copy import deepcopy


NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
if NLTK_VERSION >= version.Version("3.6.5"):
    from nltk import word_tokenize

# Uncomment to download nltk_data for the first time running.
# import nltk
# nltk.download("wordnet")
# if NLTK_VERSION >= version.Version("3.6.5"):
#     nltk.download("punkt")
# if NLTK_VERSION >= version.Version("3.6.6"):
#     nltk.download("omw-1.4")


_CITATION = """
"""

_DESCRIPTION = """\
Metric to evaluate text generation models on the grounded dialog generation task.
"""

# TODO
_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
    knowledge: task-specific grounded knowledge

Returns:
    bleu-1/2/3/4: corpus-bleu score, from sacrebleu
    rouge-1/2/L: ROUGE-F1, from rouge_score
    meteor: METEOR, from nltk
    unigram f1: unigram overlap, from parlai
    distinct-1/2: from parlai
    other knowledge utility score: task-specific knowledge utility metrics
"""

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
stop_words = set(stopwords.words("english"))
def utt2words(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    from parlai https://github.com/facebookresearch/ParlAI/blob/9daae69320c07104493486e022c0e46a7871b253/parlai/core/metrics.py#L810"""
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    return s.split()


def get_bleu(predictions, references):
    """bleu-1/2/3/4 from sacrebleu"""
    references = [" " if ref=="" else ref for ref in references]
    metrics = {}
    bleu = BLEU(lowercase=True, force=False, tokenize=BLEU.TOKENIZER_DEFAULT, smooth_method="exp", smooth_value=None, effective_order=False)
    stats = sum_of_lists(bleu._extract_corpus_statistics(predictions, [references]))
    for n in range(1,5):
        metrics[f"bleu-{n}"] = bleu.compute_bleu(
            correct=stats[2: 2 + bleu.max_ngram_order],
            total=stats[2 + bleu.max_ngram_order:],
            sys_len=int(stats[0]), ref_len=int(stats[1]),
            smooth_method=bleu.smooth_method, smooth_value=bleu.smooth_value,
            effective_order=bleu.effective_order,
            max_ngram_order=n).score
    return metrics


def get_unigram_f1(predictions, references):
    """unigram f1 between prediction and reference, from parlai"""
    metrics = {}
    metrics["unigram f1"] = []
    metrics["unigram f1 (non-stop words)"] = []
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
                metrics["unigram f1"].append(f1)
            else:
                metrics["unigram f1 (non-stop words)"].append(f1)
    metrics["unigram f1"] = np.mean(metrics["unigram f1"]) * 100
    metrics["unigram f1 (non-stop words)"] = np.mean(metrics["unigram f1 (non-stop words)"]) * 100
    return metrics


def get_rouge(predictions, references):
    """rouge-1/2/L from rouge-score"""
    rouge_types=["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for prediction, reference in zip(predictions, references):
        score = scorer.score(reference, prediction)
        aggregator.add_scores(score)

    return {key: 100 * (value.mid.fmeasure if key == "rougeL" else value.mid.recall) for key, value in aggregator.aggregate().items()}


def get_meteor(predictions, references):
    """meteor from nltk"""
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
    return {"meteor": np.mean(scores) * 100}


def get_distinct(predictions):
    """distinct-1/2 
    from parlai https://github.com/facebookresearch/ParlAI/blob/9daae69320c07104493486e022c0e46a7871b253/parlai/core/metrics.py#L781"""
    def _ngram(seq, n):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])
    
    metrics = {}
    for k in [1, 2]:
        inter_cnt = Counter()
        for prediction in predictions:
            ngram = Counter(_ngram(utt2words(prediction), k))
            inter_cnt += ngram
        metrics[f"distinct-{k}"] = max(len(inter_cnt), 1e-12) / max(sum(inter_cnt.values()), 1e-5) * 100
    return metrics


def get_nlg_slot_err(predictions, knowledge):
    """slot error rate: (missing_count + redundant_count) / all_count for value in dialog acts"""
    val2ds_dict = {}
    ontology = load_ontology("multiwoz21")
    for domain_name in ontology["domains"]:
        domain = ontology["domains"][domain_name]
        for slot_name in domain["slots"]:
            slot = domain["slots"][slot_name]
            if "possible_values" not in slot:
                continue
            possible_vals = slot["possible_values"]
            if len(possible_vals) > 0:
                for val in possible_vals:
                    val2ds_dict[val] = f"{domain_name}-{slot_name}"
    score_list = []
    for utterance, da in zip(predictions, knowledge):
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        ## missing values
        # print(da)
        # print(utterance)
        for key in ['categorical', 'non-categorical']:
            for value in da[key]['value']:
                if len(value) > 0:
                    # print(value)
                    all_values.add(value)
                    if value.strip().lower() not in utterance.lower():
                        missing_count += 1
                        # print(f"\tmissing: {value}")
                    all_count += 1
        if all_count == 0:
            continue
        ## redundant values
        for val in val2ds_dict:
            if f" {val.strip().lower()} " in f" {utterance.strip().lower()} " and val.strip().lower() not in all_values:
                wlist = val2ds_dict[val].split("-")
                domain, slot = wlist[0], wlist[1]
                if f" {slot.strip().lower()}" in f" {utterance.strip().lower()} ":
                    redundant_count += 1
                    # print(f"redundant: {val}/{val2ds_dict[val]}")
        item_score = float(missing_count + redundant_count) / all_count
        # print(f"\tredundant: {redundant_count} | missing_count: {missing_count} |all_count: {all_count}")
        # print('-'*100)
        score_list.append(item_score)
    return {"err": np.mean(score_list) * 100}


def load_entities():
    """modified (load from unified ontology) from UnifiedSKG
    https://github.com/HKUNLP/UnifiedSKG/blob/49a2ff950bb312b980c22ad72b11520db72ab6a3/metrics/kvret/evaluator.py#L8"""

    ontology = load_ontology("kvret")
    all_entities = set()
    for domain in ontology["domains"]:
        for slot in ontology["domains"][domain]["slots"]:
            all_entities |= set(ontology["domains"][domain]["slots"][slot]["possible_values"])
    missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist", "1st",
                        "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                        "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "Jill",
                        "Jack"]
    all_entities |= set(missed_entities)
    all_entities.remove("HR")
    all_entities.add(" HR ")
    all_entities = sorted(list(all_entities), key=lambda i: len(i), reverse=True)
    return all_entities


def check_sub_str(str_list: list, sub_str: str):
    """
    It takes a list of strings and a substring as input, and returns True if the substring is found
    in any of the strings in the list, and False otherwise
    """
    for str_item in str_list:
        if sub_str in str_item or sub_str.lower() in str_item.lower():
            return True
    return False


def extract_entities_from_utterance(utterance, sorted_entities):
    """modified (remove underscore) from UnifiedSKG
    https://github.com/HKUNLP/UnifiedSKG/blob/49a2ff950bb312b980c22ad72b11520db72ab6a3/metrics/kvret/response_entity_hit.py#L45"""

    utterance = " {} ".format(utterance)  # for entity matching
    for h in range(0, 13): # for formulating am & pm
        utterance = utterance.replace("{} am".format(h), "{}am".format(h))
        utterance = utterance.replace("{} pm".format(h), "{}pm".format(h))
    for entity_item_a in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for entity_item_b in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            utterance = utterance.replace("{}-{}f".format(str(entity_item_a), str(entity_item_b)), "{}f-{}f".format(str(entity_item_a), str(entity_item_b)))
    entities_in_this_utterance = []
    for entity in sorted_entities:
        # len(entity) decreases
        if (entity in utterance) or (entity.lower() in utterance.lower()):
            if not check_sub_str(entities_in_this_utterance, entity):
                # in case of "week & weekend", "week & next_week" etc
                entities_in_this_utterance.append(entity)
    return entities_in_this_utterance


def f1_score(y_pred, y_true, average="micro"):
    """micro/marco-F1 score, modified from UnifiedSKG
    https://github.com/HKUNLP/UnifiedSKG/blob/49a2ff950bb312b980c22ad72b11520db72ab6a3/metrics/kvret/response_entity_hit.py#L76"""

    assert len(y_pred) == len(y_true)

    def _compute_F1(precision, recall):
        return 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0

    def _compute_prf(gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = _compute_F1(precision, recall)
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    F1_pred, F1_count, TP_all, FP_all, FN_all = 0, 0, 0, 0, 0

    for y_true_item, y_pred_item in zip(y_true, y_pred):
        single_tp, single_fp, single_fn, single_f1, count = _compute_prf(y_true_item, y_pred_item)
        F1_pred += single_f1
        F1_count += count
        TP_all += single_tp
        FP_all += single_fp
        FN_all += single_fn

    if average == "macro":
        F1_macro_score = F1_pred / float(F1_count) if F1_count != 0 else 0
        return F1_macro_score * 100
    elif average == "micro":
        P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
        R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
        F1_micro_score = _compute_F1(P_score, R_score) * 100
        return F1_micro_score
    else:
        raise ValueError("Options other than micro/macro are not supported.")


def get_kvret_entity_f1(predictions, references, knowledge):
    """entity f1 for kvret, modified from
    https://github.com/HKUNLP/UnifiedSKG/blob/49a2ff950bb312b980c22ad72b11520db72ab6a3/metrics/kvret/response_entity_hit.py#L178"""

    global_entities = load_entities()
    F1_scores = {}
    entities_from_predictions_and_references = {
        d: {"predictions_entities": [], "references_entities": []} for d in ["all", "schedule", "weather", "navigate"]
    }
    for prediction, reference, kb in zip(predictions, references, knowledge):
        prediction_entities = extract_entities_from_utterance(utterance=prediction, sorted_entities=global_entities)
        reference_entities = extract_entities_from_utterance(utterance=reference, sorted_entities=global_entities)
        entities_from_predictions_and_references["all"]["predictions_entities"].append(prediction_entities)
        entities_from_predictions_and_references["all"]["references_entities"].append(reference_entities)
        domain = "schedule"
        for d in kb:
            if len(kb[d]["entity"]) > 0:
                domain = d
                break
        entities_from_predictions_and_references[domain]["predictions_entities"].append(prediction_entities)
        entities_from_predictions_and_references[domain]["references_entities"].append(reference_entities)
    
    for category in entities_from_predictions_and_references.keys():
        predictions_entities = entities_from_predictions_and_references[category]["predictions_entities"]
        references_entities = entities_from_predictions_and_references[category]["references_entities"]
        F1_scores["{} micro entity F1".format(category)] = f1_score(y_pred=predictions_entities, y_true=references_entities, average="micro")
        F1_scores["{} macro entity F1".format(category)] = f1_score(y_pred=predictions_entities, y_true=references_entities, average="macro")

    return {**F1_scores}


def get_opendialkg_entity_f1(predictions, references, knowledge):
    predictions_entities, references_entities = [], []
    for prediction, reference, kg_path in zip(predictions, references, knowledge):
        kg_entities = set()
        for kg_triple in kg_path:
            # add head and tail entities
            kg_entities.add(kg_triple[0])
            kg_entities.add(kg_triple[-1])
        kg_entities = sorted(list(kg_entities), key=lambda i: len(i), reverse=True)
        
        for utterance, entities in zip([prediction, reference], [predictions_entities, references_entities]):
            entities_in_this_utterance = []
            for entity in kg_entities:
                if (entity in utterance) or (entity.lower() in utterance.lower()):
                    if not check_sub_str(entities_in_this_utterance, entity):
                        # in case of "week & weekend", "week & next_week" etc
                        entities_in_this_utterance.append(entity)
            entities.append(entities_in_this_utterance)

    return {
        "micro entity f1": f1_score(y_pred=predictions_entities, y_true=references_entities, average="micro"),
        "macro entity f1": f1_score(y_pred=predictions_entities, y_true=references_entities, average="macro")
    }

def get_knowledge_sentences_f1(predictions, knowledge):
    knowledge_reference = [' '.join(k_sens) for k_sens in knowledge]
    f1_score = get_unigram_f1(predictions, knowledge_reference)
    return {f"knowledge {k}": v for k, v in f1_score.items()}


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GroundedDialogGenerationMetrics(datasets.Metric):
    """Metric to evaluate text generation models on the grounded dialog generation task."""
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("string"),
                "references": datasets.Value("string"),
                "knowledge": deepcopy(FEATURES[self.config_name]["knowledge"])
            })
        )

    def compute(self, predictions, references, knowledge=None):
        """Returns the scores: bleu"""
        metrics = {}

        # bleu
        metrics.update(get_bleu(predictions, references))
                
        # unigram f1
        metrics.update(get_unigram_f1(predictions, references))
        
        # rouge-1/2/L-fmeasure
        metrics.update(get_rouge(predictions, references))

        # meteor
        metrics.update(get_meteor(predictions, references))

        # inter-distinct-1/2
        metrics.update(get_distinct(predictions))
        
        if knowledge is not None:
            if self.config_name == "nlg":
                metrics.update(get_nlg_slot_err(predictions, knowledge))
            elif self.config_name == "kvret":
                metrics.update(get_kvret_entity_f1(predictions, references, knowledge))
            elif self.config_name == "opendialkg":
                metrics.update(get_opendialkg_entity_f1(predictions, references, knowledge))
            elif self.config_name in ["wow", "personachat"]:
                metrics.update(get_knowledge_sentences_f1(predictions, knowledge))

        return metrics
