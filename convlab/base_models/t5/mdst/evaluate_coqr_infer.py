import json
from pprint import pprint
import datasets
import os


def evaluate(args):
    dataset = datasets.load_dataset("json", data_files=args.predict_result)['train']
    
    origin_dials = json.load(open(args.origin_dials))
    metric = {'total': 0, 'diff': 0}
    for sample in dataset:
        metric['total'] += 1
        output = sample['output']
        prediction = sample['predictions']
        if prediction != output and sample['turn_idx']>0:
            metric['diff'] += 1
            dial = origin_dials[sample['dial_idx']]
            dial['turns'][sample['turn_idx']]['ori_utterance'] = dial['turns'][sample['turn_idx']]['utterance']
            dial['turns'][sample['turn_idx']]['utterance'] = prediction

    metric['diff'] = "%.2f" % (metric['diff']/metric['total'])

    filename = os.path.join(args.data_dir, os.path.basename(args.origin_dials).replace('domain', 'domain_coqr'))
    json.dump(origin_dials, open(filename, 'w', encoding='utf-8'), indent=2)
        
    return metric

def evaluate_chatgpt(args):
    # chatgpt_dials = datasets.load_dataset("json", data_files=args.predict_result)['train']
    chatgpt_dials = []
    with open(args.predict_result) as f:
        for line in f:
            chatgpt_dials.append(json.loads(line.strip()))

    origin_dials = json.load(open(args.origin_dials))
    metric = {'total': 0, 'diff': 0}
    assert len(chatgpt_dials) == len(origin_dials)
    for chatgpt_dial, dial in zip(chatgpt_dials, origin_dials):
        for turn_idx in range(0, len(dial['turns']),2):
            metric['total'] += 1
            turn = dial['turns'][turn_idx]
            assert turn['utterance'] == chatgpt_dial['turns'][turn_idx]['utterance']
            assert 'chatgpt_utterance' in chatgpt_dial['turns'][turn_idx]
            chatgpt_utt = chatgpt_dial['turns'][turn_idx]['chatgpt_utterance']
            if chatgpt_utt != turn['utterance'] and turn_idx > 0:
                metric['diff'] += 1
                turn['ori_utterance'] = turn['utterance']
                turn['utterance'] = chatgpt_utt

    metric['diff'] = "%.2f" % (metric['diff']/metric['total'])

    filename = os.path.join(args.data_dir, os.path.basename(args.origin_dials).replace('domain', 'domain_coqr_chatgpt'))
    json.dump(origin_dials, open(filename, 'w', encoding='utf-8'), indent=2)

    return metric

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file')
    parser.add_argument('--origin_dials', '-i', type=str, default=None, help='path to the original dialog')
    parser.add_argument('--data_dir', '-d', type=str, default=None, help='path to the processed sample')
    args = parser.parse_args()
    print(args)
    if 'chatgpt' in args.predict_result:
        metrics = evaluate_chatgpt(args)
    else:
        metrics = evaluate(args)
    pprint(metrics)
