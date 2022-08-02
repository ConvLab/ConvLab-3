import os
import json
from tqdm import tqdm
from convlab.util import load_dataset, load_unified_data, load_nlu_data

def create_nlg_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker='system', use_context=True, context_window_size=3)
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = [(turn['speaker'], turn['utterance']) for turn in sample['context']]
            response = sample['utterance']
            if len(context) > 0 and len(response) > 0:
                knowledge = sample['dialogue_acts']
                data.append(json.dumps({'context': context, 'knowledge': knowledge, 'response': response}, ensure_ascii=False)+'\n')

        if 'test' in data_split:
            file_name = os.path.join(os.path.dirname(data_dir), f"{data_split}.json")
        else:
            file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_kvret_data(dataset, data_dir, args):
    data_by_split = load_unified_data(dataset, speaker='system', utterance=True, db_results=True, use_context=True, context_window_size=100)
    os.makedirs(data_dir, exist_ok=True)

    domain2entity_col = {'schedule': 'event' ,'navigate': 'poi', 'weather': 'location'}
    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = [(turn['speaker'], turn['utterance']) for turn in sample['context']]
            response = sample['utterance']
            if len(context) > 0 and len(response) > 0:
                knowledge = sample['db_results']
                for domain, db_items in knowledge.items():
                    entity_col = domain2entity_col[domain]
                    for db_item in db_items:
                        db_item['entity'] = db_item.pop(entity_col)

                data.append(json.dumps({'context': context, 'knowledge': knowledge, 'response': response}, ensure_ascii=False)+'\n')

        if 'test' in data_split:
            file_name = os.path.join(os.path.dirname(data_dir), f"{data_split}.json")
        else:
            file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_personachat_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for dial in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            knowledge = dial['persona']['system']
            context = []
            for turn in dial['turns']:
                response = turn['utterance']
                if turn['speaker'] == 'system' and len(context) > 0 and len(response) > 0:
                    data.append(json.dumps({'context': context, 'knowledge': knowledge, 'response': response}, ensure_ascii=False)+'\n')
                context.append((turn['speaker'], turn['utterance']))

        if 'test' in data_split:
            file_name = os.path.join(os.path.dirname(data_dir), f"{data_split}.json")
        else:
            file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_wow_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)
    data_by_split['test'] = data_by_split['test_seen'] + data_by_split['test_unseen']
    data_by_split.pop('test_seen')
    data_by_split.pop('test_unseen')

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for dial in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = []
            for turn in dial['turns']:
                response = turn['utterance']
                if turn['speaker'] == 'system' and len(context) > 0 and len(response) > 0:
                    knowledge = turn['checked_passage']
                    if knowledge is None:
                        knowledge = []
                    elif isinstance(knowledge, str):
                        knowledge = [knowledge]
                    data.append(json.dumps({'context': context, 'knowledge': knowledge, 'response': response}, ensure_ascii=False)+'\n')
                context.append((turn['speaker'], turn['utterance']))

        if 'test' in data_split:
            file_name = os.path.join(os.path.dirname(data_dir), f"{data_split}.json")
        else:
            file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_opendialkg_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for dial in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = []
            for turn in dial['turns']:
                response = turn['utterance']
                if turn['speaker'] == 'system' and 'kg_path' in turn and len(context) > 0 and len(response) > 0:
                    knowledge = turn['kg_path']['triples']
                    data.append(json.dumps({'context': context, 'knowledge': knowledge, 'response': response}, ensure_ascii=False)+'\n')
                context.append((turn['speaker'], turn['utterance']))

        if 'test' in data_split:
            file_name = os.path.join(os.path.dirname(data_dir), f"{data_split}.json")
        else:
            file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['nlg', 'kvret', 'opendialkg', 'personachat', 'wow'], help='names of tasks')
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--shot', '-s', type=float, default=None, help='how many data is used for training and evaluation, ratio if < 1 else absolute number')
    parser.add_argument('--dial_ids_order', '-o', type=int, default=None, help='which data order is used for experiments')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name, dial_ids_order=args.dial_ids_order)
        if args.shot:
            if args.shot < 1:
                dataset['train'] = dataset['train'][:round(len(dataset['train'])*args.shot)]
                dataset['validation'] = dataset['validation'][:round(len(dataset['validation'])*args.shot)]
            else:
                args.shot = int(args.shot)
                dataset['train'] = dataset['train'][:args.shot]
                dataset['validation'] = dataset['validation'][:args.shot]
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join('data', task_name, (dataset_name if not args.shot else f'{dataset_name}_{args.shot}shot_order{args.dial_ids_order}'))
            data_by_split = eval(f"create_{task_name}_data")(dataset, data_dir, args)
