import sys
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
sys.path.append('../..')
import json
from pprint import pprint
from evaluate_util import GentScorer
from convlab2.util.unified_datasets_util import load_ontology
import numpy as np

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

def evaluate(predict_result, ontology):
    predict_result = json.load(open(predict_result))
    metrics = {}

    # BLEU Score
    evaluator = GentScorer()
    references = []
    candidates = []
    for i in range(len(predict_result)):
        references.append([word_tokenize(predict_result[i]['utterance'])])
        candidates.append(word_tokenize(predict_result[i]['prediction']))
    metrics['bleu'] = corpus_bleu(references, candidates)

    # ERROR Rate
    ## get all values in ontology
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
    for item in predict_result:
        da = item['dialogue_acts']
        utterance = item['prediction']
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        ## missing values
        for key in da:
            slot_value = da[key]
            for triple in slot_value:
                if 'value' in triple:
                    value = triple['value']
                    all_values.add(value)
                    if value.strip().lower() not in utterance.lower():
                        missing_count += 1
                        # logger.log(f"missing: {triple['slot']}-{triple['value']} | {item['prediction']} | {item['utterance']}")
                    all_count += 1
        if all_count == 0:
            continue
        ## redundant values
        for val in val2ds_dict:
            if f' {val.strip().lower()} ' in f' {utterance.strip().lower()} ' and val.strip().lower() not in all_values:
                wlist = val2ds_dict[val].split('-')
                domain, slot = wlist[0], wlist[1]
                if f' {slot.strip().lower()}' in f' {utterance.strip().lower()} ':
                    redundant_count += 1
                    # logger.log(f"redundant: {val}/{val2ds_dict[val]} | {item['prediction']} | {item['utterance']}")
        item_score = float(missing_count + redundant_count) / all_count
        # logger.log(f"redundant: {redundant_count} | missing_count: {missing_count} |all_count: {all_count}")
        score_list.append(item_score)
    metrics['err'] = np.mean(score_list)

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="calculate NLG metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True,
                        help='path to the prediction file that in the unified data format')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='the name of the dataset to be evaluated')
    args = parser.parse_args()
    print(args)
    ontology = load_ontology(args.dataset_name)
    # logger = Logging('./evaluate_unified_datasets.log')
    metrics = evaluate(args.predict_result, ontology)
    pprint(metrics)
