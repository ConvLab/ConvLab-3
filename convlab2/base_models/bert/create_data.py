import os
import json
from tqdm import tqdm
from convlab2.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from collections import Counter

def create_bio_data(dataset, data_dir):
    data_by_split = load_nlu_data(dataset, speaker='all')
    os.makedirs(data_dir, exist_ok=True)

    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    data_splits = data_by_split.keys()
    cnt = Counter()
    for data_split in data_splits:
        data = []
        for sample in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            utterance = sample['utterance']
            dialogue_acts = [da for da in sample['dialogue_acts']['non-categorical'] if 'start' in da]
            cnt[len(dialogue_acts)] += 1

            sentences = sent_tokenizer.tokenize(utterance)
            sent_spans = sent_tokenizer.span_tokenize(utterance)
            tokens = [token for sent in sentences for token in word_tokenizer.tokenize(sent)]
            token_spans = [(sent_span[0]+token_span[0], sent_span[0]+token_span[1]) for sent, sent_span in zip(sentences, sent_spans) for token_span in word_tokenizer.span_tokenize(sent)]
            labels = ['O'] * len(tokens)
            for da in dialogue_acts:
                char_start = da['start']
                char_end = da['end']
                word_start, word_end = -1, -1
                for i, token_span in enumerate(token_spans):
                    if char_start == token_span[0]:
                        word_start = i
                    if char_end == token_span[1]:
                        word_end = i + 1
                if word_start == -1 and word_end == -1:
                    # char span does not match word, skip
                    continue
                labels[word_start] = 'B'
                for i in range(word_start+1, word_end):
                    labels[i] = "I"
            data.append(json.dumps({'tokens': tokens, 'labels': labels}, ensure_ascii=False)+'\n')
        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
    print('num of spans in utterances', cnt)

def create_dialogBIO_data(dataset, data_dir):
    data_by_split = load_nlu_data(dataset, split_to_turn=False)
    os.makedirs(data_dir, exist_ok=True)

    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()

    data_splits = data_by_split.keys()
    cnt = Counter()
    for data_split in data_splits:
        data = []
        for dialog in tqdm(data_by_split[data_split], desc=f'{data_split} sample', leave=False):
            all_tokens, all_labels = [], []
            for sample in dialog['turns']:
                speaker = sample['speaker']
                utterance = sample['utterance']
                dialogue_acts = [da for da in sample['dialogue_acts']['non-categorical'] if 'start' in da]
                cnt[len(dialogue_acts)] += 1

                sentences = sent_tokenizer.tokenize(utterance)
                sent_spans = sent_tokenizer.span_tokenize(utterance)
                tokens = [token for sent in sentences for token in word_tokenizer.tokenize(sent)]
                token_spans = [(sent_span[0]+token_span[0], sent_span[0]+token_span[1]) for sent, sent_span in zip(sentences, sent_spans) for token_span in word_tokenizer.span_tokenize(sent)]
                labels = ['O'] * len(tokens)
                for da in dialogue_acts:
                    char_start = da['start']
                    char_end = da['end']
                    word_start, word_end = -1, -1
                    for i, token_span in enumerate(token_spans):
                        if char_start == token_span[0]:
                            word_start = i
                        if char_end == token_span[1]:
                            word_end = i + 1
                    if word_start == -1 and word_end == -1:
                        # char span does not match word, skip
                        continue
                    labels[word_start] = 'B'
                    for i in range(word_start+1, word_end):
                        labels[i] = "I"
                all_tokens.extend([speaker, ':']+tokens)
                all_labels.extend(['O', 'O']+labels)
            data.append(json.dumps({'tokens': all_tokens, 'labels': all_labels}, ensure_ascii=False)+'\n')
        file_name = os.path.join(data_dir, f"{data_split}.json")
        with open(file_name, "w", encoding='utf-8') as f:
            f.writelines(data)
    print('num of spans in utterances', cnt)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', metavar='task_name', nargs='*', choices=['bio', 'dialogBIO'], help='names of tasks')
    parser.add_argument('--datasets', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--save_dir', metavar='save_directory', type=str, default='data', help='directory to save the data, default: data/$task_name/$dataset_name')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join(args.save_dir, task_name, dataset_name)
            eval(f"create_{task_name}_data")(dataset, data_dir)
