import os
import json
from tqdm import tqdm
from convlab2.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data

def create_rg_data(data_by_split, data_dir):
    data_splits = data_by_split.keys()
    file_name = os.path.join(data_dir, f"source_prefix.txt")
    with open(file_name, "w") as f:
        f.write("generate a system response according to the context: ")
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = ' '.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']])
            response = f"{sample['speaker']}: {sample['utterance']}"
            data.append(json.dumps({'context': context, 'response': response})+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w") as f:
            f.writelines(data)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', metavar='task_name', nargs='*', choices=['rg'], help='names of tasks')
    parser.add_argument('--datasets', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--save_dir', metavar='save_directory', type=str, default='data', help='directory to save the data, default: data/$task_name/$dataset_name')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_by_split = eval(f"load_{task_name}_data")(dataset)
            data_dir = os.path.join(args.save_dir, task_name, dataset_name)
            os.makedirs(data_dir, exist_ok=True)
            eval(f"create_{task_name}_data")(data_by_split, data_dir)