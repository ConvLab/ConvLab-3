import json
from pprint import pprint


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {'TP':0, 'FP':0, 'FN':0}
    acc = []
    for sample in predict_result:
        pred_state = sample['predictions']['state']
        gold_state = sample['state']
        flag = True
        for domain in gold_state:
            for slot, values in gold_state[domain].items():
                if domain not in pred_state or slot not in pred_state[domain]:
                    predict_values = ''
                else:
                    predict_values = ''.join(pred_state[domain][slot].split()).lower()
                if len(values) > 0:
                    if len(predict_values) > 0:
                        values = [''.join(value.split()).lower() for value in values.split('|')]
                        predict_values = [''.join(value.split()).lower() for value in predict_values.split('|')]
                        if any([value in values for value in predict_values]):
                            metrics['TP'] += 1
                        else:
                            metrics['FP'] += 1
                            metrics['FN'] += 1
                            flag = False
                    else:
                        metrics['FN'] += 1
                        flag = False
                else:
                    if len(predict_values) > 0:
                        metrics['FP'] += 1
                        flag = False

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
