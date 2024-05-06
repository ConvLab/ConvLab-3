import json
import os
from pprint import pprint
import re
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from convlab.base_models.t5.mdst.utils import get_value_from_bio_tags, filter_rewrite
from convlab.base_models.t5.mdst.evaluate_qa import eval_slot_pairs_prediction


def evaluate(args):
    dataset = datasets.load_dataset("json", data_files=args.predict_result)['train']

    bio_tagger = pipeline(task="token-classification", model="../../bert/output/bio/tm1+tm2+tm3", aggregation_strategy="simple", device=0)
    if "predictions" in dataset[0]:
        source_key = "output"
        target_key = "predictions"
    elif "chatgpt_utterance" in dataset[0]:
        source_key = "utterance"
        target_key = "chatgpt_utterance"
    output_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, source_key), batch_size=64)), total=len(dataset)):
        output_values.append(get_value_from_bio_tags(dataset[idx][source_key], out))
    dataset = dataset.add_column('output_values', output_values)
    predictions_values = []
    for idx, out in tqdm(zip(range(len(dataset)),bio_tagger(KeyDataset(dataset, target_key), batch_size=64)), total=len(dataset)):
        predictions_values.append(get_value_from_bio_tags(dataset[idx][target_key], out))
    dataset = dataset.add_column('predictions_values', predictions_values)
    del bio_tagger

    if args.recover:
        recover_prompt = 'Rewrite the question to remove anaphora and make it self-contained according to the given context.'
        recover_inputs = []
        for sample in dataset:
            context = sample['input'].split('\n\n')[1:-1]
            recover_inputs.append('\n\n'.join([recover_prompt]+context+[f'question: {sample["predictions"]}']))
        dataset = dataset.add_column('recover_input', recover_inputs)
        recover_utterances = []
        recover_model = pipeline(task="text2text-generation", model="output/canard/origin", device=0)
        for idx, out in tqdm(zip(range(len(dataset)),recover_model(KeyDataset(dataset, 'recover_input'), batch_size=32, max_length=100)), total=len(dataset)):
            recover_utterances.append(out[0]['generated_text'])
        dataset = dataset.add_column('recover_predictions', recover_utterances)
        dataset = dataset.remove_columns(['recover_input'])
        del recover_model
    
    origin_dials = json.load(open(args.origin_dials))
    rewrite_dials = []
    metric = {'total': 0, 'diff': 0, 'filtered': 0}
    replaced_samples = []
    coqr_slot_pairs = {}
    for sample in dataset:
        metric['total'] += 1
        output = sample[source_key]
        prediction = sample[target_key]
        if target_key == "chatgpt_utterance":
            values = [item['source'][-2] for item in sample['replace_state']]
        else:
            values = sample['src_values']
        if prediction != output:
            metric['diff'] += 1
            recover_predictions = None if 'recover_predictions' not in sample else sample['recover_predictions']
            if filter_rewrite(output, prediction, values, sample['output_values'], sample['predictions_values'], recover_predictions):
                metric['filtered'] += 1
                replaced_samples.append(json.dumps(sample)+'\n')
                dial = origin_dials[sample['dial_idx']]
                dial['turns'][sample['turn_idx']]['coqr_utterance'] = prediction
                if args.recover:
                    dial['turns'][sample['turn_idx']]['recover'] = recover_predictions
                rewrite_dials.append(dial)
                assert "replace_state" in dial['turns'][sample['turn_idx']]
                for slot_pair in dial['turns'][sample['turn_idx']]['replace_state']:
                    src_d, src_s, _ = slot_pair['source']
                    tgt_d, tgt_s, _ = slot_pair['target']
                    k = str((src_d, src_s, tgt_d, tgt_s))
                    coqr_slot_pairs.setdefault(k, 0)
                    coqr_slot_pairs[k] += 1
    json.dump(coqr_slot_pairs, open(os.path.abspath(args.origin_dials+'/..'+'/coqr_slot_pairs.json'), 'w'), indent=2)
    gold_slot_pairs = json.load(open(os.path.abspath(args.origin_dials+'/..'*3+'/multi_domain_slot_pairs.json')))
    precision, recall, f1 = eval_slot_pairs_prediction(set(gold_slot_pairs), set(coqr_slot_pairs))
    print(f'coqr vs gold, precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}')
    qadst_slot_pairs = json.load(open(os.path.abspath(args.origin_dials+'/..'*2+'/qadst_slot_pairs.json')))
    precision, recall, f1 = eval_slot_pairs_prediction(set(qadst_slot_pairs), set(coqr_slot_pairs))
    print(f'coqr vs qadst, precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}')


    metric['diff'] = "%.2f" % (metric['diff']/metric['total'])
    metric['filtered'] = "%.2f" % (metric['filtered']/metric['total'])
    
    output_file = args.predict_result.replace('generated_predictions', 'filtered_predictions')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(replaced_samples)

    if target_key == 'predictions':
        output_file = args.origin_dials.replace('aug_dials', f'aug_dials_coqr')
    elif target_key == "chatgpt_utterance":
        output_file = args.origin_dials.replace('aug_dials', f'aug_dials_chatgpt')
    json.dump(rewrite_dials, open(output_file, 'w', encoding='utf-8'), indent=2)
        
    return metric

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--origin_dials', '-o', type=str, required=True, help='path to the dialogs from AUG_TYPE_CONCAT2REL')
    parser.add_argument('--recover', '-r', action='store_true', help='whether use recover rewrite as filter')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
