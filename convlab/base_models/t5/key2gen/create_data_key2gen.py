import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from convlab.util import load_dataset, load_unified_data, load_nlu_data
from convlab.base_models.t5.nlu.serialization import serialize_dialogue_acts
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation

def create_nlg_data(dataset, data_dir, args):
    data_by_split = load_nlu_data(dataset, speaker=args.speaker, use_context=args.context_window_size>0, context_window_size=args.context_window_size)
    data_dir = os.path.join(data_dir, args.speaker, f'context_{args.context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            if args.key2gen:
                keywords = []
                for da_type in sample['dialogue_acts']:
                    for da in sample['dialogue_acts'][da_type]:
                        intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
                        intent_domain = f'{intent}-{domain}'
                        keywords.append([intent_domain])
                        if len(slot) > 0:
                            keywords[-1].append(slot)
                        if len(value) > 0:
                            keywords[-1].append(value)
                dialogue_acts_seq = '| {} |'.format(' | '.join([' : '.join(da_keywords) for da_keywords in keywords]))
            else:
                dialogue_acts_seq = serialize_dialogue_acts(sample['dialogue_acts'])

            if args.context_window_size>0:
                context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
                context = f'generate a response: grounded knowledge: {dialogue_acts_seq} context:\n\n{context}'
            else:
                context = f'generate a response: grounded knowledge: {dialogue_acts_seq} context:\n\n{sample["speaker"]}: '

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
                # DONE: try adding prompt, no improvement
                entity_cnt = Counter()
                for triple in triples:
                    e1, r, e2 = triple
                    for e in [e1, e2]:
                        if e.startswith('[') and e.endswith(']'):
                            continue
                        entity_cnt[e] += 1
                        
                assert len(entity_cnt) > 0
                common_entity = entity_cnt.most_common(1)[0][0]
                context = f'{" | ".join(keywords)}\n\ncontext: user: tell me something about {common_entity}. system: '
            else:
                triples = [' : '.join(triple) for triple in triples]
                context = f'{" | ".join(triples)}\n\ncontext: system: '

            data.append(json.dumps({'triples': context, 'text': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_commongen_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            sample = sample['turns'][0]
            concepts = sample['concepts']
            context = f'{" | ".join(concepts)}\n\ncontext: system: '

            data.append(json.dumps({'concepts': context, 'text': sample['utterance']}, ensure_ascii=False)+'\n')

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
            if len(sample['utterance']) == 0:
                continue
            db_results = sample['db_results']
            db_seqs = []
            for domain, db_items in db_results.items():
                entity_col = domain2entity_col[domain]
                for db_item in db_items:
                    entity = db_item[entity_col]
                    for db_key, db_value in db_item.items():
                        if db_key == entity_col:
                            continue
                        db_seqs.append(' : '.join([entity, db_key, db_value]))
            db_seq = ' |\n'.join(db_seqs)

            context = '\n'.join([f"{turn['speaker']}: {turn['utterance']}" for turn in sample['context']]+[f'{sample["speaker"]}: '])
            context = f'generate a response: all knowledge:\n\n| {db_seq} | context:\n\n{context}'
            data.append(json.dumps({'context+db': context, 'response': sample['utterance']}, ensure_ascii=False)+'\n')

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_personachat_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    stop_words = set(stopwords.words('english')) | set(punctuation)
    def sentence2keywords(sentence):
        index2keyword = {}
        for i, w in enumerate(word_tokenize(sentence)):
            if not w.lower() in stop_words:
                index2keyword[i] = w
        indexes = sorted(index2keyword.keys())
        keywords = []
        for i, index in enumerate(indexes):
            if i > 0 and index == indexes[i-1] + 1:
                keywords[-1]+= ' '+index2keyword[index]
            else:
                keywords.append(index2keyword[index])
        return keywords

    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for dial in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            persona = dial['persona']['system']
            if args.key2gen:
                persona_seq = ' |\n'.join([' : '.join(sentence2keywords(s)) for s in persona])
            else:
                persona_seq = ' | '.join(persona)
            context = []
            for turn in dial['turns']:
                if turn['speaker'] == 'system':
                    context_seq = '\n'.join([f"{t['speaker']}: {t['utterance']}" for t in context]+[f'{turn["speaker"]}: '])
                    context_seq = f'generate a response: all knowledge:\n\n| {persona_seq} | context:\n\n{context_seq}'
                    data.append(json.dumps({'context+persona': context_seq, 'response': turn['utterance']}, ensure_ascii=False)+'\n')
                context.append({'speaker': turn['speaker'], 'utterance': turn['utterance']})

        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
        data_by_split[data_split] = data
    return data_by_split

def create_wow_data(dataset, data_dir, args):
    data_by_split = dataset
    os.makedirs(data_dir, exist_ok=True)

    stop_words = set(stopwords.words('english')) | set(punctuation)
    def sentence2keywords(sentence):
        index2keyword = {}
        for i, w in enumerate(word_tokenize(sentence)):
            if not w.lower() in stop_words:
                index2keyword[i] = w
        indexes = sorted(index2keyword.keys())
        keywords = []
        for i, index in enumerate(indexes):
            if i > 0 and index == indexes[i-1] + 1:
                keywords[-1]+= ' '+index2keyword[index]
            else:
                keywords.append(index2keyword[index])
        return keywords

    def sentences2keywords_seq(sentences):
        return ' |\n'.join([' : '.join(sentence2keywords(sentence)) for sentence in sent_tokenize(sentences)])


    data_splits = data_by_split.keys()
    for data_split in data_splits:
        data = []
        for dial in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            context = []
            for turn in dial['turns']:
                if turn['speaker'] == 'system':
                    if turn['checked_sentence']:
                        if args.key2gen:
                            know_seq = f" | {sentences2keywords_seq(turn['checked_sentence'])} |"
                        else:
                            know_seq = turn['checked_sentence']
                    else:
                        know_seq = ''
                    context_seq = '\n'.join([f"{t['speaker']}: {t['utterance']}" for t in context]+[f'{turn["speaker"]}: '])
                    context_seq = f'generate a response: grounded knowledge:\n\n{know_seq} context:\n\n{context_seq}'
                    data.append(json.dumps({'context+knowledge': context_seq, 'response': turn['utterance']}, ensure_ascii=False)+'\n')
                context.append({'speaker': turn['speaker'], 'utterance': turn['utterance']})

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
    parser.add_argument('--tasks', '-t', metavar='task_name', nargs='*', choices=['nlg', 'dart', 'commongen', 'kvret', 'personachat', 'wow'], help='names of tasks')
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
