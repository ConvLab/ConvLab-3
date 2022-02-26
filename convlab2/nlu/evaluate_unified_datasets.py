import json
from pprint import pprint


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {x: {'TP':0, 'FP':0, 'FN':0} for x in ['overall', 'binary', 'categorical', 'non-categorical']}

    for sample in predict_result:
        for da_type in ['binary', 'categorical', 'non-categorical']:
            if da_type == 'binary':
                predicts = [(x['intent'], x['domain'], x['slot']) for x in sample['predictions']['dialogue_acts'][da_type]]
                labels = [(x['intent'], x['domain'], x['slot']) for x in sample['dialogue_acts'][da_type]]
            else:
                predicts = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in sample['predictions']['dialogue_acts'][da_type]]
                labels = [(x['intent'], x['domain'], x['slot'], ''.join(x['value'].split()).lower()) for x in sample['dialogue_acts'][da_type]]
            for ele in predicts:
                if ele in labels:
                    metrics['overall']['TP'] += 1
                    metrics[da_type]['TP'] += 1
                else:
                    metrics['overall']['FP'] += 1
                    metrics[da_type]['FP'] += 1
            for ele in labels:
                if ele not in predicts:
                    metrics['overall']['FN'] += 1
                    metrics[da_type]['FN'] += 1
    
    for metric in metrics:
        TP = metrics[metric].pop('TP')
        FP = metrics[metric].pop('FP')
        FN = metrics[metric].pop('FN')
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        metrics[metric]['precision'] = precision
        metrics[metric]['recall'] = recall
        metrics[metric]['f1'] = f1

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
