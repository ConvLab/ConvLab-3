import json
import os
from pprint import pprint
import re
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from convlab.base_models.t5.mdst.utils import get_value_from_bio_tags, filter_rewrite


def remove_values_chunks(entity_groups, sentence, values):
    group2rm = set()
    for value in values:
        try:
            value_start_idx = sentence.lower().index(value.lower())
        except:
            return sentence
        value_end_idx = value_start_idx+len(value)
        for idx, group in enumerate(entity_groups):
            if value_start_idx < group['end'] and value_end_idx > group['start']:
                group2rm.add(idx)
                if idx < len(entity_groups)-1 and entity_groups[idx+1]['entity_group'] == 'PP':
                    group2rm.add(idx+1)
                    if idx+1 < len(entity_groups)-1 and entity_groups[idx+2]['entity_group'] == 'NP':
                        group2rm.add(idx+2)
                if idx > 0 and entity_groups[idx-1]['entity_group'] != 'NP':
                    group2rm.add(idx-1)
                break
    for idx in sorted(list(group2rm), reverse=True):
        start, end = entity_groups[idx]['start'], entity_groups[idx]['end']
        if (idx - 1) in group2rm:
            entity_groups[idx-1]['end'] = end
        else:
            sentence = sentence[:start] + sentence[end:]
    return ' '.join(sentence.split())


def evaluate(args):
    value_tagger = pipeline(task="token-classification", model=args.model_name_or_path, aggregation_strategy="simple", device=0)
    dataset = datasets.load_dataset("json", data_files=args.predict_result)['train']
    new_utterances = []
    for idx, out in tqdm(zip(range(len(dataset)),value_tagger(KeyDataset(dataset, "utterance"), batch_size=64)), total=len(dataset)):
        new_utterances.append(remove_values_chunks(out, dataset[idx]['utterance'], dataset[idx]['src_values']))
    dataset = dataset.add_column('new_utterance', new_utterances)
    del value_tagger

    bio_tagger = pipeline(task="token-classification", model="../../bert/output/bio/tm1+tm2+tm3", aggregation_strategy="simple", device=0)
    ori_utterance_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, "utterance"), batch_size=64)), total=len(dataset)):
        ori_utterance_values.append(get_value_from_bio_tags(dataset[idx]['utterance'], out))
    dataset = dataset.add_column('ori_utterance_values', ori_utterance_values)
    new_utterance_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, "new_utterance"), batch_size=64)), total=len(dataset)):
        new_utterance_values.append(get_value_from_bio_tags(dataset[idx]['new_utterance'], out))
    dataset = dataset.add_column('new_utterance_values', new_utterance_values)
    
    origin_dials = json.load(open(args.origin_dials))
    rewrite_dials = []
    metric = {'total': 0, 'diff': 0, 'filtered': 0}
    replaced_samples = []
    for sample in dataset:
        metric['total'] += 1
        utterance = sample['utterance']
        new_utterance = sample['new_utterance']
        values = sample['src_values']
        if new_utterance != utterance:
            metric['diff'] += 1
            if filter_rewrite(utterance, new_utterance, values, sample['ori_utterance_values'], sample['new_utterance_values']):
                metric['filtered'] += 1
                replaced_samples.append(json.dumps(sample)+'\n')
                dial = origin_dials[sample['dial_idx']]
                dial['turns'][sample['turn_idx']]['elli_utterance'] = new_utterance
                rewrite_dials.append(dial)

    metric['diff'] = "%.2f" % (metric['diff']/metric['total'])
    metric['filtered'] = "%.2f" % (metric['filtered']/metric['total'])
    
    output_file = args.predict_result.replace('elli', 'elli_filtered_predictions')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(replaced_samples)

    aug_times = args.origin_dials.split('/')[-2].split('_')[-1]
    filename = args.origin_dials.split('/')[-1].replace('aug_dials', f'aug_dials_elli_{aug_times}')
    output_file = os.path.join(os.path.dirname(os.path.dirname(args.origin_dials)), filename)
    json.dump(rewrite_dials, open(output_file, 'w', encoding='utf-8'), indent=2)
        
    return metric

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--model_name_or_path', '-m', type=str, required=True, help='path to the value tagger model')
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--origin_dials', '-o', type=str, required=True, help='path to the dialogs from AUG_TYPE_CONCAT2REL')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
