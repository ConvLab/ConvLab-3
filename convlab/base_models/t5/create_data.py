import os
import json
from tqdm import tqdm
import re
from transformers import AutoTokenizer
from convlab.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data, retrieve_utterances
from convlab.base_models.t5.nlu.serialization import serialize_dialogue_acts, deserialize_dialogue_acts, equal_da_seq
from convlab.base_models.t5.dst.serialization import serialize_dialogue_state, deserialize_dialogue_state, equal_state_seq

def create_rg_data(dataset, data_dir, args):
    data_by_split = load_rg_data(dataset, speaker=args.speaker)
    data_dir = os.path.join(data_dir, args.speaker)
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            if len(sample['context']) == 0:
                continue
            context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
            data.append(json.dumps({'context': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_nlu_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))
            data.append(json.dumps({'context': context, 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_dst_data(dataset, data_dir, args):
    data_by_split = load_dst_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            for domain in sample['state']:
                for slot in sample['state'][domain]:
                    vs = sample['state'][domain][slot].split('|')
                    # only the first variation of value
                    sample['state'][domain][slot] = vs[0]
            state_seq = serialize_dialogue_state(sample['state'])
            assert equal_state_seq(sample['state'], state_seq), print(sample['state'], state_seq, deserialize_dialogue_state(state_seq))
            data.append(json.dumps({'context': context, 'state_seq': state_seq}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_nlg_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            if len(dialogue_acts_seq) == 0:
                # skip empty dialogue acts
                continue
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
                context = f'{dialogue_acts_seq}\n\n{context}'
            else:
                context = f'{dialogue_acts_seq}\n\n{sample["speaker"]}: '
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))
            data.append(json.dumps({'context+da': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_goal2dialogue_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            goal = re.sub(r'<.*?>', '', sample['goal']['description'])
            dialogue = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['turns']])
            data.append(json.dumps({'goal': goal, 'dialogue': dialogue}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_retnlu_data(dataset, data_dir, args):
    dataset_name = dataset[list(dataset.keys())[0]][0]['dataset']
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}', \
        f'in_context_{args.retrieval_in_context}', f'topk_{args.retrieval_topk}')
    os.makedirs(data_dir, exist_ok=True)

    turn_pool = []
    for d in args.retrieval_datasets:
        pool_dataset = load_dataset(d)
        for turn in load_nlu_data(pool_dataset, data_split='train', speaker=args.speaker)['train']:
            if any([len(das) > 0 for da_type, das in turn['dialogue_acts'].items()]):
                turn_pool.append({'dataset': d, **turn})

    data_splits = data_by_split.keys()
    query_turns = []
    for data_split in data_splits:
        query_turns.extend(data_by_split[data_split])
    augmented_dataset = retrieve_utterances(query_turns, turn_pool, args.retrieval_topk, 'all-MiniLM-L6-v2')

    i = 0
    for data_split in data_splits:
        data = []
        for j in tqdm(range(len(data_by_split[data_split])), desc=f'{data_split} sample', leave=False):
            sample = augmented_dataset[i+j]
            response = f"{sample['speaker']}: {sample['utterance']}"
            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            context = ' '.join([dataset_name, context])
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))

            retrieved_turns = sample['retrieved_turns']
            for t in retrieved_turns:
                # in-context learning
                retrieved_utterance = f"{t['dataset']} {t['speaker']}: {t['utterance']}"
                retrieved_dialogue_acts_seq = serialize_dialogue_acts(t['dialogue_acts'])
                if args.retrieval_in_context:
                    context = f"{retrieved_utterance} => {retrieved_dialogue_acts_seq}\n\n" + context
                elif data_split != 'test':
                    data.append(json.dumps({'context': retrieved_utterance, 'dialogue_acts_seq': retrieved_dialogue_acts_seq}, ensure_ascii=False)+'\n')        

            data.append(json.dumps({'context': context, 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')
        i += len(data_by_split[data_split])

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
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['rg', 'nlu', 'dst', 'nlg', 'goal2dialogue', 'retnlu', 'retnlg'], help='names of tasks')
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--speaker', '-s', type=str, choices=['user', 'system', 'all'], help='speaker(s)')
    parser.add_argument('--context_window_size', '-c', type=int, default=0, help='how many contextual utterances are considered')
    parser.add_argument('--len_tokenizer', '-l', type=str, default=None, help='name or path of tokenizer that used to get seq len')
    parser.add_argument('--ratio', '-r', type=float, default=None, help='how many data is used for training and evaluation')
    parser.add_argument('--dial_ids_order', '-o', type=int, default=None, help='which data order is used for experiments')
    parser.add_argument('--retrieval_datasets', metavar='dataset_name for retrieval augmentation', nargs='*', help='names of unified datasets for retrieval')
    parser.add_argument('--retrieval_topk', type=int, default=3, help='how many utterances to be retrieved')
    parser.add_argument('--retrieval_in_context', action='store_true', default=False, help='whether use the retrieved utterance by in-context learning')
    args = parser.parse_args()
    print(args)
    if args.len_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.len_tokenizer)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        if args.ratio:
            dataset = load_dataset(dataset_name, dial_ids_order=args.dial_ids_order, split2ratio={'train': args.ratio, 'validation': args.ratio})
        else:
            dataset = load_dataset(dataset_name, args.dial_ids_order)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join('data', task_name, (dataset_name if not args.ratio else f'{dataset_name}_{args.ratio}_order{args.dial_ids_order}'))
            data_by_split = eval(f"create_{task_name}_data")(dataset, data_dir, args)
            if args.len_tokenizer:
                get_max_len(data_by_split, tokenizer)
