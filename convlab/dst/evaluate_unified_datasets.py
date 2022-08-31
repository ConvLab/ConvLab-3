import json
import os
from pprint import pprint

import numpy as np


def load_results(predict_results):
    files = [file.strip() for file in predict_results.split(',')]
    files = [file for file in files if os.path.isfile(file)]

    predictions = []
    for file in files:
        reader = open(file, 'r')
        predictions += json.load(reader)
        reader.close()

    return predictions


def evaluate(predict_result):
    predict_result = load_results(predict_result)

    metrics = {'TP': 0, 'FP': 0, 'FN': 0}
    jga = []
    aga = []
    fga = []
    l2_err = []
    lamb = [0.25, 0.5, 0.75, 1.0]

    for sample in predict_result:
        pred_state = sample['predictions']['state']
        gold_state = sample['state']
        utt_idx = sample['utt_idx']

        predicts = {(domain, slot, ''.join(value.split()).lower()) for domain in pred_state
                    for slot, value in pred_state[domain].items() if value}
        labels = {(domain, slot, ''.join(value.split()).lower()) for domain in gold_state
                  for slot, value in gold_state[domain].items() if value}
        predicts, labels = sorted(list(predicts)), sorted(list(labels))

        # Flexible goal accuracy (see https://arxiv.org/pdf/2204.03375.pdf)
        weighted_err = [1] * len(lamb)
        if utt_idx == 0:
            err_idx = -999999
            predicts_prev = []
            labels_prev = []

            if predicts != labels:
                err_idx = utt_idx
                weighted_err = [0] * len(lamb)
        else:
            if predicts != labels:
                predicts_changes = [ele for ele in predicts if ele not in predicts_prev]
                labels_changes = [ele for ele in labels if ele not in labels_prev]

                new_predict_err = [ele for ele in predicts_changes if ele not in labels]
                new_predict_miss = [ele for ele in labels_changes if ele not in predicts]

                if new_predict_err or new_predict_miss:
                    weighted_err = [0] * len(lamb)
                    err_idx = utt_idx
                else:
                    err_age = utt_idx - err_idx
                    weighted_err = [1 - np.exp(-l * err_age) for l in lamb]
            predicts_prev = predicts
            labels_prev = labels
        fga.append(weighted_err)

        _l2 = 2.0 * len([ele for ele in labels if ele not in predicts])
        _l2 += 2.0 * len([ele for ele in predicts if ele not in labels])
        l2_err.append(_l2)

        flag = True
        for ele in predicts:
            if ele in labels:
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        for ele in labels:
            if ele not in predicts:
                metrics['FN'] += 1
        flag &= (predicts == labels)
        jga.append(flag)
    
    TP = metrics.pop('TP')
    FP = metrics.pop('FP')
    FN = metrics.pop('FN')
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    metrics[f'slot_f1'] = f1
    metrics[f'slot_precision'] = precision
    metrics[f'slot_recall'] = recall
    metrics['joint_goal_accuracy'] = sum(jga) / len(jga)
    for i, l in enumerate(lamb):
        metrics[f'flexible_goal_accuracy_{l}'] = sum(weighted_err[i] for weighted_err in fga)/len(fga)
    metrics['l2_norm_error'] = sum(l2_err) / len(l2_err)

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
