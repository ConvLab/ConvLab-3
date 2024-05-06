import json
import os
from pprint import pprint
from copy import deepcopy
import pandas as pd

class StatCount:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def add(self, golden, pred):
        self.data.append((golden, pred))

    def compute(self):
        acc = []
        mat = [[[],[]],[[],[]]]
        TP, FP, FN = 0, 0, 0
        for golden, pred in self.data:
            if golden == '<no answer>':
                # golden == none
                if 'no answer' in pred:
                    # prediction == none
                    # assert pred == 'no answer>', print(sample)
                    flag = True
                    mat[1][1].append(flag)
                else:
                    # prediction != none
                    flag = False
                    FP += 1
                    mat[0][1].append(flag)
            else:
                # golden != none
                if 'no answer' in pred:
                    # prediction == none
                    # assert pred == 'no answer>'
                    flag = False
                    FN += 1
                    mat[1][0].append(flag)
                else:
                    # prediction != none
                    if golden == pred:
                        # prediction == golden
                        flag = True
                        TP += 1
                    else:
                        flag = False
                        FP += 1
                        FN += 1
                    mat[0][0].append(flag)
            acc.append(flag)
        
        acc = sum(acc)/len(acc) if len(acc) > 0 else 0
        precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
        recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        for i in range(2):
            for j in range(2):
                mat[i][j] = (len(mat[i][j]), sum(mat[i][j])/len(mat[i][j]) if len(mat[i][j]) > 0 else 0)
        df = pd.DataFrame({'non-empty golden': [mat[0][0], mat[1][0]], 'empty golden': [mat[0][1], mat[1][1]]}, index=['non-empty pred', 'empty pred'])
        return acc, f1, df, recall, TP


def eval_slot_pairs_prediction(golden_slot_pairs: set, predict_slot_pairs: set):
    TP = len(golden_slot_pairs & predict_slot_pairs)
    FP = len(predict_slot_pairs) - TP
    FN = len(golden_slot_pairs) - TP
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, f1


def filter_by_true_cross_domain(golden_slot_pairs: set, predict_slot_pairs: dict) -> dict:
    cross_domain = set()
    for slot_pair in golden_slot_pairs:
        src_d, src_s, dst_d, dst_s = eval(slot_pair)
        cross_domain.add((src_d, dst_d))
    filtered_slot_pairs = {}
    for slot_pair in predict_slot_pairs:
        src_d, src_s, dst_d, dst_s = eval(slot_pair)
        if (src_d, dst_d) in cross_domain:
            filtered_slot_pairs[slot_pair] = predict_slot_pairs[slot_pair]
    return filtered_slot_pairs


def filter_single_domain_dials(single_domain_dials, slot_pairs):
    filtered_dials = []
    for dial in single_domain_dials:
        dial = deepcopy(dial)
        assert len(dial['domains']) == 1
        src_domain = dial['domains'][0]
        real_state = dial['turns'][-2]['state']
        if src_domain not in real_state:
            continue
        predict_state = dial.pop('qadst')
        state = {}
        for domain in predict_state:
            if domain == src_domain:
                continue
            state[domain] = {}
            for slot, value in predict_state[domain].items():
                if 'no answer' not in value:
                    state[domain][slot] = value
            if len(state[domain]) == 0:
                state.pop(domain)
        
        for dst_domain in state:
            assert dst_domain != src_domain
            for dst_slot, dst_value in state[dst_domain].items():
                for src_slot, src_value in real_state[src_domain].items():
                    slot_pair_key = str((src_domain, src_slot, dst_domain, dst_slot))
                    if slot_pair_key not in slot_pairs:
                        continue
                    if dst_value.strip().lower() == src_value.strip().lower():
                        dial.setdefault('qa', {})
                        dial['qa'].setdefault(dst_domain, {})
                        dial['qa'][dst_domain][dst_slot] = [src_slot, src_value]
        filtered_dials.append(dial)

    return filtered_dials


def evaluate(args):
    true_cross_domain_slot_pairs = json.load(open(os.path.join(args.data_dir, 'multi_domain_slot_pairs.json')))
    full_state = json.load(open(os.path.join(args.data_dir, 'full_state.json')))
    slot_pairs = {(d1, s1, d2, s2): StatCount() 
                for d1 in full_state for s1 in full_state[d1] 
                for d2 in full_state if d2 != d1 for s2 in full_state[d2]}

    for data_split in ['train', 'validation']:
        original_filename = os.path.join(args.data_dir, f'{data_split}_single_domain.json')
        prediction_filename = os.path.join(args.predict_dir, f'{data_split}_single_domain_qa_generated_predictions.json')
        single_domain_dials = json.load(open(original_filename))            

        state = deepcopy(full_state)
        stat_count = StatCount()
        with open(prediction_filename) as predict_result:
            samples = [json.loads(sample) for sample in predict_result]
            for sample_idx, sample in enumerate(samples):
                dial_idx = sample['dial_idx']

                src_domain, dst_domain = sample['domains']
                dst_slot = sample['slot']
                dst_value = sample['predictions']

                state[dst_domain][dst_slot] = dst_value
                    
                if dst_domain == src_domain:
                    # single domain slot acc/f1
                    stat_count.add(sample['output'], sample['predictions'])

                if sample_idx == len(samples)-1 or dial_idx != samples[sample_idx+1]['dial_idx']:
                    if data_split == 'train':
                        for src_slot in state[src_domain]:
                            src_value = state[src_domain][src_slot]
                            for domain in full_state:
                                if domain != src_domain:
                                    for slot in state[domain]:
                                        value = state[domain][slot]
                                        slot_pairs[(src_domain, src_slot, domain, slot)].add(src_value, value)
                    single_domain_dials[dial_idx]['qadst'] = state
                    state = deepcopy(full_state)
        
        output_file = prediction_filename.replace('generated_predictions', 'result').replace('.json', '.md')
        with open(output_file, 'w', encoding='utf-8') as f:
            acc, f1, metrics, _, _ = stat_count.compute()
            print('acc', acc, file=f)
            print('f1', f1, file=f)
            print(metrics, file=f)
        
        if data_split == 'train':
            for slot_pair, stat_cnt in slot_pairs.items():
                acc, f1, _, recall, TP = stat_cnt.compute()
                slot_pairs[slot_pair] = [f1, recall, TP]
            
            total_slot_pairs = len(slot_pairs)
            slot_pairs = {str(k):v for k,v in sorted(slot_pairs.items(),key=lambda x: -x[1][0]) if v[-1] > 1}
            with open(os.path.join(args.data_dir, 'qadst_slot_pairs.json'), "w", encoding='utf-8') as f:
                json.dump(slot_pairs, f, indent=2)

            slot_pairs_dict = {}
            print('total slot pairs', total_slot_pairs)
            for f1_th in [0, 0.01, 0.1]:
                filtered_slot_pairs = dict(filter(lambda x: x[1][0]>f1_th, slot_pairs.items()))
                precision, recall, f1 = eval_slot_pairs_prediction(set(true_cross_domain_slot_pairs.keys()), set(filtered_slot_pairs.keys()))
                print(f'f1_th>{f1_th}, precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}, size={len(filtered_slot_pairs)}, filtered_ratio={len(filtered_slot_pairs)/total_slot_pairs:.2f}')
                slot_pairs_dict[f'f1_th>{f1_th}'] = filtered_slot_pairs

            filtered_slot_pairs = filter_by_true_cross_domain(set(true_cross_domain_slot_pairs.keys()), slot_pairs_dict['f1_th>0.1'])
            precision, recall, f1 = eval_slot_pairs_prediction(set(true_cross_domain_slot_pairs.keys()), set(filtered_slot_pairs.keys()))
            print(f'f1_th>0.1_true_domain_comb, precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}, size={len(filtered_slot_pairs)}, filtered_ratio={len(filtered_slot_pairs)/total_slot_pairs:.2f}')
            slot_pairs_dict[f'f1_th>0.1_true_domain_comb'] = filtered_slot_pairs

            slot_pairs_dict['true_slot_pairs'] = true_cross_domain_slot_pairs

        for k, filtered_slot_pairs in slot_pairs_dict.items():
            sub_dir = os.path.join(args.data_dir, f'qadst_{k}')
            if data_split == 'train':
                os.makedirs(sub_dir, exist_ok=True)
                output_file = os.path.join(sub_dir, 'qadst_slot_pairs.json')
                json.dump(filtered_slot_pairs, open(output_file, 'w', encoding='utf-8'), indent=2)
            filtered_dials = filter_single_domain_dials(single_domain_dials, filtered_slot_pairs)
            output_file = os.path.join(sub_dir, os.path.basename(original_filename))
            json.dump(filtered_dials, open(output_file, 'w', encoding='utf-8'), indent=2)
    
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--predict_dir', '-p', type=str, required=True, help='path to the output dir containing prediction file')
    parser.add_argument('--data_dir', '-d', type=str, default=None, help='path to the dataset dir containing full_state.json and multi_domain_slot_pairs.json')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
