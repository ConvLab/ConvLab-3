import os
import json
from tqdm import tqdm
import re
from convlab2.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data
from convlab2.base_models.t5.nlu.serialization import serialize_dialogue_acts, deserialize_dialogue_acts, equal_da_seq

def create_rg_data(dataset, data_dir, args):
    data_by_split = load_rg_data(dataset, speaker=args.speaker)
    data_dir = os.path.join(data_dir, args.speaker)
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = ' '.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']])
            response = f"{sample['speaker']}: {sample['utterance']}"
            data.append(json.dumps({'context': context, 'response': response}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)

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
                context = ' '.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[response])
            else:
                context = response
            dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])
            assert equal_da_seq(sample['dialogue_acts'], dialogue_acts_seq), print(sample['dialogue_acts'], dialogue_acts_seq, deserialize_dialogue_acts(dialogue_acts_seq))
            data.append(json.dumps({'context': context, 'dialogue_acts_seq': dialogue_acts_seq}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)

def create_goal2dialogue_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            goal = re.sub(r'<.*?>', '', sample['goal']['description'])
            dialogue = ' '.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['turns']])
            data.append(json.dumps({'goal': goal, 'dialogue': dialogue}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['rg', 'nlu', 'goal2dialogue'], help='names of tasks')
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--speaker', '-s', type=str, choices=['user', 'system', 'all'], help='speaker(s)')
    parser.add_argument('--context_window_size', '-c', type=int, default=0, help='how many contextual utterances are considered')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join('data', task_name, dataset_name)
            eval(f"create_{task_name}_data")(dataset, data_dir, args)
