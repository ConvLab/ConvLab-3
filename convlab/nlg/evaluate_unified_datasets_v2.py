import sys
import sacrebleu
import json
from pprint import pprint
from convlab.util.unified_datasets_util import load_ontology
import numpy as np


sys.path.append('../..')
int2word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'}
logger = None


class Logging:
    def __init__(self, path):
        file = open(path, 'w+')
        file.write('')
        file.close()
        self.path = path

    def log(self, sent):
        with open(self.path, 'a') as f:
            f.write(sent)
            f.write('\n')
            f.close()


def evaluate(predict_result, ontology, filter_empty_acts=True):
    predict_result = json.load(open(predict_result))
    metrics = {}

    # BLEU Score
    references = []
    candidates = []
    for i in range(len(predict_result)):
        if filter_empty_acts:
            acts = predict_result[i]['dialogue_acts']
            acts_size = len(
                acts['binary']) + len(acts['categorical']) + len(acts['non-categorical'])
            if acts_size == 0:
                continue
        references.append(predict_result[i]['utterance'])
        if 'prediction' in predict_result[i]:
            candidates.append(predict_result[i]['prediction'])
        else:
            candidates.append(predict_result[i]['predictions']['utterance'])
    # metrics['bleu'] = corpus_bleu(references, candidates)
    references = [" " if ref == "" else ref for ref in references]
    metrics['bleu'] = sacrebleu.corpus_bleu(
        candidates, [references], lowercase=True).score

    # ERROR Rate
    # get all values in ontology
    val2ds_dict = {}
    for domain_name in ontology['domains']:
        domain = ontology['domains'][domain_name]
        for slot_name in domain['slots']:
            slot = domain['slots'][slot_name]
            if 'possible_values' not in slot:
                continue
            possible_vals = slot['possible_values']
            if len(possible_vals) > 0:
                for val in possible_vals:
                    val2ds_dict[val] = f'{domain_name}-{slot_name}'

    score_list = []
    total_count = 0
    total_missing = 0
    total_hallucination = 0
    for item in predict_result:
        da = item['dialogue_acts']
        utterance = item['predictions']['utterance'] if 'predictions' in item else item['prediction']
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        # missing values
        for key in da:
            slot_value = da[key]
            for triple in slot_value:
                if 'value' in triple:
                    value = triple['value']
                    all_values.add(value)
                    missing_flag = False
                    norm_value = value.strip().lower()
                    if norm_value not in utterance.lower():
                        missing_flag = True
                        # # problem 1: values with multiple tokens considered missing if some spaces are dropped in the nlg output
                        # # e.g. "53 - 57 Lensfield road" considered missing despite "53-57 Lensfield road" in the output
                        # # temporary solution: for values with multiple tokens, remove spaces in the value and the sentence before comparing
                        # if len(norm_value.split()) > 1:
                        #     if norm_value.replace(' ', '') in utterance.lower().replace(' ', ''):
                        #         missing_flag = False
                        # problem 2: integer values considered missing if they are realised in words
                        # e.g. "4 stars" considered missing despite "four stars" in the output
                        # temporary solution: use a dictionary (int2word) for value matching
                        if norm_value in int2word:
                            if int2word[norm_value] in utterance.lower():
                                missing_flag = False
                        # problem 3: misspelt values.
                        # problem 3.1: truly misspelt: "huntingdon marriot hotel" vs "huntingdon marriott hotel"
                        # problem 3.2: British vs American (probably some other usage) English: "caffe" vs "cafe", "theatre" vs "theater"
                        # ignore for now
                    if missing_flag:
                        missing_count += 1
                        logger.log(
                            f"missing: {triple['slot']}-{triple['value']} | {item['prediction']} | {item['utterance']}")
                    all_count += 1
        if all_count == 0:
            continue
        # redundant values
        for val in val2ds_dict:
            # problem 1: the checked value from other domain-slot is a substring of one of the values in the dialogue action
            # e.g. centre vs centre area
            mentioned_flag = False
            for mentioned_value in all_values:
                if val.strip().lower() in mentioned_value.strip().lower():
                    mentioned_flag = True
            if f' {val.strip().lower()} ' in f' {utterance.strip().lower()} ' and not mentioned_flag:
                wlist = val2ds_dict[val].split('-')
                domain, slot = wlist[0], wlist[1]
                redundant_flag = False
                norm_slot = slot.strip().lower()
                if f' {norm_slot}' in f' {utterance.strip().lower()} ':
                    redundant_flag = True
                    # # problem 2: binary slots not checked
                    # if norm_slot in [da['slot'] for da in da['binary']]:
                    #     redundant_flag = False
                    # problem 3: for the dataset, missing annotation in dialogue_acts
                if redundant_flag:
                    redundant_count += 1
                    logger.log(f"{all_values}")
                    logger.log(
                        f"redundant: {val}/{val2ds_dict[val]} | {item['prediction']} | {item['utterance']}")
        item_score = float(missing_count + redundant_count) / all_count
        # logger.log(f"redundant: {redundant_count} | missing_count: {missing_count} |all_count: {all_count}")
        score_list.append(item_score)
        total_missing += missing_count
        total_hallucination += redundant_count
        total_count += all_count
    metrics['err'] = np.mean(score_list)
    metrics['missing'] = total_missing
    metrics['redundant'] = total_hallucination
    metrics['total'] = total_count

    return metrics


def ser_new(dialog_acts, utts, filter_empty_acts=True):
    ontology = load_ontology('multiwoz21')
    metrics = {}

    # ERROR Rate
    # get all values in ontology
    val2ds_dict = {}
    for domain_name in ontology['domains']:
        domain = ontology['domains'][domain_name]
        for slot_name in domain['slots']:
            slot = domain['slots'][slot_name]
            if 'possible_values' not in slot:
                continue
            possible_vals = slot['possible_values']
            if len(possible_vals) > 0:
                for val in possible_vals:
                    val2ds_dict[val] = f'{domain_name}-{slot_name}'

    score_list = []
    total_count = 0
    total_missing = 0
    total_hallucination = 0
    # for item in predict_result:
    for utterance, da in zip(utts, dialog_acts):
        # da = dialog_act
        # utterance = utt
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        # missing values
        for idsv in da:
            i, d, s, v = [x.lower() for x in idsv]
            # slot_value = da[key]
            # for triple in slot_value:
            #     if 'value' in triple:
            if v != 'none':
                all_values.add(v)
                missing_flag = False
                norm_value = v.strip().lower()
                if norm_value not in utterance.lower():
                    missing_flag = True
                    # problem 1: values with multiple tokens considered missing if some spaces are dropped in the nlg output
                    # e.g. "53 - 57 Lensfield road" considered missing despite "53-57 Lensfield road" in the output
                    # temporary solution: for values with multiple tokens, remove spaces in the value and the sentence before comparing
                    if len(norm_value.split()) > 1:
                        if norm_value.replace(' ', '') in utterance.lower().replace(' ', ''):
                            missing_flag = False
                    # problem 2: integer values considered missing if they are realised in words
                    # e.g. "4 stars" considered missing despite "four stars" in the output
                    # temporary solution: use a dictionary (int2word) for value matching
                    if norm_value in int2word:
                        if int2word[norm_value] in utterance.lower():
                            missing_flag = False
                    # problem 3: misspelt values.
                    # problem 3.1: truly misspelt: "huntingdon marriot hotel" vs "huntingdon marriott hotel"
                    # problem 3.2: British vs American (probably some other usage) English: "caffe" vs "cafe", "theatre" vs "theater"
                    # ignore for now
                if missing_flag:
                    missing_count += 1
                    # logger.log(f"missing: {s}-{v} | {utterance}")
                all_count += 1
            if all_count == 0:
                continue
        # redundant values
        for val in val2ds_dict:
            # problem 1: the checked value from other domain-slot is a substring of one of the values in the dialogue action
            # e.g. centre vs centre area
            mentioned_flag = False
            for mentioned_value in all_values:
                if val.strip().lower() in mentioned_value.strip().lower():
                    mentioned_flag = True
            if f' {val.strip().lower()} ' in f' {utterance.strip().lower()} ' and not mentioned_flag:
                wlist = val2ds_dict[val].split('-')
                domain, slot = wlist[0], wlist[1]
                redundant_flag = False
                norm_slot = slot.strip().lower()
                if f' {norm_slot}' in f' {utterance.strip().lower()} ':
                    redundant_flag = True
                    # # problem 2: binary slots not checked - not applicable for old format
                    # if norm_slot in [da['slot'] for da in da['binary']]:
                    #     redundant_flag = False
                    # problem 3: for the dataset, missing annotation in dialogue_acts
                if redundant_flag:
                    redundant_count += 1
                    # logger.log(f"{all_values}")
                    # logger.log(f"redundant: {val}/{val2ds_dict[val]} | {utterance}")
        # item_score = float(missing_count + redundant_count) / all_count
        # logger.log(f"redundant: {redundant_count} | missing_count: {missing_count} |all_count: {all_count}")
        # score_list.append(item_score)
        total_missing += missing_count
        total_hallucination += redundant_count
        total_count += all_count
    # metrics['err'] = np.mean(score_list)
    metrics['missing'] = total_missing
    metrics['redundant'] = total_hallucination
    metrics['total'] = total_count

    return total_missing, total_hallucination, total_count, None, None


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="calculate NLG metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True,
                        help='path to the prediction file that in the unified data format')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='the name of the dataset to be evaluated')
    args = parser.parse_args()
    print(args)
    ontology = load_ontology(args.dataset_name)
    logger = Logging('./evaluate_unified_datasets.log')
    metrics = evaluate(args.predict_result, ontology)
    pprint(metrics)
