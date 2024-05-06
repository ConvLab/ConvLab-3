import json
import os
from pprint import pprint
import re
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from convlab.base_models.t5.mdst.utils import get_value_from_bio_tags, filter_rewrite


def evaluate(args):
    bio_tagger = pipeline(task="token-classification", model="../../bert/output/bio/tm1+tm2+tm3", aggregation_strategy="simple", device=0)
    dataset = datasets.load_dataset("json", data_files=args.predict_result)['train']
    output_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, "output"), batch_size=64)), total=len(dataset)):
        output_values.append(get_value_from_bio_tags(dataset[idx]['output'], out))
    dataset = dataset.add_column('output_values', output_values)
    predictions_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, "predictions"), batch_size=64)), total=len(dataset)):
        predictions_values.append(get_value_from_bio_tags(dataset[idx]['predictions'], out))
    dataset = dataset.add_column('predictions_values', predictions_values)
    
    origin_dials = json.load(open(args.origin_dials))
    rewrite_dials = []
    metric = {'total': 0, 'diff': 0, 'filtered': 0}
    replaced_samples = []
    for sample in dataset:
        metric['total'] += 1
        output = sample['output']
        prediction = sample['predictions']
        values = sample['src_values']
        if prediction != output:
            metric['diff'] += 1
            if filter_rewrite(output, prediction, values, sample['output_values'], sample['predictions_values']):
                metric['filtered'] += 1
                replaced_samples.append(json.dumps(sample)+'\n')
                dial = origin_dials[sample['dial_idx']]
                dial['turns'][sample['turn_idx']]['coqr_utterance'] = prediction
                rewrite_dials.append(dial)

    metric['diff'] = "%.2f" % (metric['diff']/metric['total'])
    metric['filtered'] = "%.2f" % (metric['filtered']/metric['total'])
    
    output_file = args.predict_result.replace('generated_predictions', 'filtered_predictions')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(replaced_samples)

    aug_times = args.origin_dials.split('/')[-2].split('_')[-1]
    filename = args.origin_dials.split('/')[-1].replace('aug_dials', f'aug_dials_coqr_{aug_times}')
    output_file = os.path.join(os.path.dirname(os.path.dirname(args.origin_dials)), filename)
    json.dump(rewrite_dials, open(output_file, 'w', encoding='utf-8'), indent=2)
        
    return metric

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--origin_dials', '-o', type=str, required=True, help='path to the dialogs from AUG_TYPE_CONCAT2REL')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
