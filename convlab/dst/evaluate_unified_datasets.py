import json
from pprint import pprint


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {'TP':0, 'FP':0, 'FN':0}
    acc = []

    for sample in predict_result:
        pred_state = sample['predictions']['state']
        gold_state = sample['state']
        predicts = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in pred_state for slot, value in pred_state[domain].items() if len(value)>0}))
        labels = sorted(list({(domain, slot, ''.join(value.split()).lower()) for domain in gold_state for slot, value in gold_state[domain].items() if len(value)>0}))

        flag = True
        for ele in predicts:
            if ele in labels:
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        for ele in labels:
            if ele not in predicts:
                metrics['FN'] += 1
        flag &= (predicts==labels)
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

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
