import os
import json
from pprint import pprint


def evaluate(args):
    original_dials = json.load(open(args.origin_data))

    gold_pred = []
    with open(args.predict_result) as predict_result:
        samples = [json.loads(sample) for sample in predict_result]
        for sample_idx, sample in enumerate(samples):
            dial_idx = sample['dial_idx']
            turn_idx = sample['turn_idx']
            # predict_state_update = deserialize_dialogue_state(sample['predictions'])
            # predict_active_domains = list(predict_state_update.keys())
            predict_active_domains = sample['predictions'].split(';')

            if len(predict_active_domains) == 0 and \
                sample_idx != 0 and dial_idx == samples[sample_idx-1]['dial_idx'] and turn_idx == samples[sample_idx-1]['turn_idx']+2:
                # inherit prev active domains
                predict_active_domains = original_dials[dial_idx]['turns'][turn_idx-2]['predict_active_domains']

            gold_pred.append((sample['active_domains'], predict_active_domains))
            original_dials[dial_idx]['turns'][turn_idx]['predict_active_domains'] = predict_active_domains

    acc = []
    TP, FP, FN = 0, 0, 0
    for gold, pred in gold_pred:
        acc.append(sorted(gold) == sorted(pred))
        TP += len(set(gold)&set(pred))
        FP += len(set(pred)-set(gold))
        FN += len(set(gold)-set(pred))
    acc = sum(acc)/len(acc)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    output_file = args.predict_result.replace('generated_predictions', 'result').replace('.json', '.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        print(metrics, file=f)

    output_filename = os.path.join(args.data_dir, os.path.basename(args.origin_data))
    json.dump(original_dials, open(output_filename, 'w', encoding='utf-8'), indent=2)
    
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate active domain prediction metrics")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--origin_data', '-i', type=str, default=None, help='path to the original dialog')
    parser.add_argument('--data_dir', '-d', type=str, default=None, help='path to the processed sample')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
