import os
import json
from tqdm import tqdm
import re
from transformers import AutoTokenizer
from convlab2.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data
from convlab2.base_models.t5.nlu.serialization import serialize_dialogue_acts, deserialize_dialogue_acts, equal_da_seq
from collections import Counter

def create_nlg_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            if args.key2gen:
                da_dict = {}
                for da_type in sample['dialogue_acts']:
                    for da in sample['dialogue_acts'][da_type]:
                        intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
                        intent_domain = f'{intent}-{domain}'
                        da_dict.setdefault(intent_domain, [])
                        da_dict[intent_domain].append((slot, value))
                keywords = []
                for intent_domain, slot_values in da_dict.items():
                    keywords.append(intent_domain)
                    for slot, value in slot_values:
                        if len(slot) > 0:
                            keywords.append(slot)
                        if len(value) > 0:
                            keywords.append(value)
                dialogue_acts_seq = ' | '.join(keywords)
            else:
                dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])

            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
                context = f'{dialogue_acts_seq}\n\ncontext: {context}'
            else:
                context = f'{dialogue_acts_seq}\n\ncontext: {sample["speaker"]}: '
            data.append(json.dumps({'context+da': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_dart_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            sample = sample['turns'][0]
            triples = sample['tripleset']
            if args.key2gen:
                keywords = [w for triple in triples for w in triple]
                # TODO: try adding prompt
                # entity_cnt = Counter()
                # for triple in triples:
                #     e1, r, e2 = triple
                #     for e in [e1, e2]:
                #         if e.startswith('[') and e.endswith(']'):
                #             continue
                #         entity_cnt[e] += 1
                        
                # assert len(entity_cnt) > 0
                # common_entity = entity_cnt.most_common(1)[0][0]
                # context = f'{" | ".join(keywords)}\n\ncontext: user: tell me something about {common_entity}. system: '

                context = f'{" | ".join(keywords)}\n\ncontext: system: '
            else:
                triples = [f"[{triple[0]}][{triple[1]}][{triple[2]}]" for triple in triples]
                context = f'{";".join(triples)}\n\ncontext: system: '

            data.append(json.dumps({'triples': context, 'text': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split


def get_max_len(data_by_split, tokenizer):
    for data_split in data_by_split.keys():
        seq_len = {}
        for line in data_by_split[data_split]:
            item = json.loads(line.strip())
            for column, seq in item.items():
                seq_len.setdefault(column, [])
                seq_len[column].append(len(tokenizer.tokenize(seq)))
        print(f"data split: {data_split}")
        for column, lens in seq_len.items():
            print(f'\t{column}\tmax_len: {max(lens)}\tmean_len: {round(sum(lens)/len(lens),2)}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['nlg', 'dart'], help='names of tasks')
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--speaker', '-s', type=str, choices=['user', 'system', 'all'], help='speaker(s)')
    parser.add_argument('--context_window_size', '-c', type=int, default=0, help='how many contextual utterances are considered')
    parser.add_argument('--len_tokenizer', '-l', type=str, default=None, help='name or path of tokenizer that used to get seq len')
    parser.add_argument('--ratio', '-r', type=float, default=None, help='how many data is used for training and evaluation')
    parser.add_argument('--dial_ids_order', '-o', type=int, default=None, help='which data order is used for experiments')
    parser.add_argument('--key2gen', '-k', action='store_true', default=False, help='generate data for key2gen models')
    args = parser.parse_args()
    print(args)
    if args.len_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.len_tokenizer)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name, args.dial_ids_order)
        if args.ratio:
            dataset['train'] = dataset['train'][:round(len(dataset['train'])*args.ratio)]
            dataset['validation'] = dataset['validation'][:round(len(dataset['validation'])*args.ratio)]
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            if args.key2gen:
                data_dir = os.path.join('data', task_name, "key2gen_"+(dataset_name if not args.ratio else f'{dataset_name}_{args.ratio}_order{args.dial_ids_order}'))
            else:
                data_dir = os.path.join('data', task_name, (dataset_name if not args.ratio else f'{dataset_name}_{args.ratio}_order{args.dial_ids_order}'))
            data_by_split = eval(f"create_{task_name}_data")(dataset, data_dir, args)
            if args.len_tokenizer:
                get_max_len(data_by_split, tokenizer)
