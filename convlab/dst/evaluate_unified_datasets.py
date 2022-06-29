import json
from pprint import pprint

import numpy as np


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {'TP':0, 'FP':0, 'FN':0}
    acc = []
    fga = []
    lamb = [0.25, 0.5, 0.75, 1.0]

    for sample in predict_result:
        pred_state = sample['predictions']['state']
        gold_state = sample['state']
        utt_idx = sample['utt_idx']
        predicts = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in pred_state for slot, value in pred_state[domain].items() if len(value)>0}))
        labels = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in gold_state for slot, value in gold_state[domain].items() if len(value)>0}))

        w = [1] * len(lamb)
        if utt_idx == 0:
            err_idx = -999999
            predicts_prev = []
            labels_prev = []

            if predicts != labels:
                err_idx = utt_idx
                w = [0] * len(lamb)
        else:
            if predicts != labels:
                predicts_changes = [ele for ele in predicts if ele not in predicts_prev]
                labels_changes = [ele for ele in labels if ele not in labels_prev]

                new_predict_err = [ele for ele in predicts_changes if ele not in labels]
                new_predict_miss = [ele for ele in labels_changes if ele not in predicts]

                if new_predict_err or new_predict_miss:
                    w = [0] * len(lamb)
                    err_idx = utt_idx
                else:
                    x = utt_idx - err_idx
                    w = [1 - np.exp(-l * x) for l in lamb]
            predicts_prev = predicts
            labels_prev = labels
        fga.append(w)

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
        acc.append(flag)
    
    TP = metrics.pop('TP')
    FP = metrics.pop('FP')
    FN = metrics.pop('FN')
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    metrics[f'slot_f1'] = f1
    metrics[f'slot_precision'] = precision
    metrics[f'slot_recall'] = recall
    metrics['accuracy'] = sum(acc)/len(acc)
    for i, l in enumerate(lamb):
        metrics[f'flexible_goal_accuracy_{l}'] = sum(a[i] for a in fga)/len(fga)

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
