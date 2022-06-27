import os
import json
from tqdm import tqdm
from convlab.util import load_dataset, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data, load_rg_data
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from collections import Counter
import json_lines
from convlab.util.unified_datasets_util import create_delex_data

def create_bio_data(dataset, data_dir, args):
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

def create_dialogBIO_data(dataset, data_dir, args):
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

def create_revert_dialogBIO_data(dataset, data_dir, args):
    def tag2da(tokens, tags):
        assert len(tokens)==len(tags)
        triples = []
        i = 0
        utt = ''
        while i < len(tags):
            tag = tags[i]
            if tag == 'B':
                value = tokens[i]
                j = i + 1
                while j < len(tags):
                    next_tag = tags[j]
                    if next_tag == 'I':
                        value += ' ' + tokens[j]
                        i += 1
                        j += 1
                    else:
                        break
                triples.append({'intent':'', 'domain':'', 'slot':'', 'value': value, 'start': len(utt), 'end': len(utt)+len(value)})
                utt += value + ' '
                assert utt[triples[-1]['start']:triples[-1]['end']] == value, print(utt[triples[-1]['start']:triples[-1]['end']],triples[-1])
            else:
                utt += tokens[i] + ' '
            i += 1
        utt = utt[:-1]
        assert utt == ' '.join(tokens), print(utt, '\n', ' '.join(tokens))
        return triples

    def dialog2turn(tokens, labels):
        turns = []
        turn = {'tokens': [], 'tags': []}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if i < len(tokens) - 1 and token in ['user', 'system'] and tokens[i+1] == ':':
                turns.append(turn)
                turn = {'tokens': [], 'tags': []}
                i += 2
                continue
            turn['tokens'].append(token)
            turn['tags'].append(labels[i])
            i += 1
        turns.pop(0)
        for turn in turns:
            da = {'binary': [], 'categorical': [], 'non-categorical': []}
            da['non-categorical'] = tag2da(turn['tokens'], turn['tags'])
            turn['utterance'] = ' '.join(turn['tokens'])
            turn['dialogue_acts'] = da
        return turns

    for data_split in dataset:
        infer_output_data_path = os.path.join(args.infer_data_dir, f'{data_split}.json')
        for original_dial, bio_dial in zip(dataset[data_split], json_lines.reader(open(infer_output_data_path))):
            bio_turns = dialog2turn(bio_dial['tokens'], bio_dial['labels'])
            original_dial['turns'] = original_dial['turns'][:len(bio_turns)]
            assert len(bio_turns) == len(original_dial['turns']), print(len(bio_turns), len(original_dial['turns']))
            for ori_turn, new_turn in zip(original_dial['turns'], bio_turns):
                ori_turn['original_utterance'] = ori_turn['utterance']
                ori_turn['utterance'] = new_turn['utterance']
                ori_turn['original_dialogue_acts'] = ori_turn['dialogue_acts']
                ori_turn['dialogue_acts'] = new_turn['dialogue_acts']
    dataset, _ = create_delex_data(dataset, delex_func=lambda d,s,v: f'<v>{v}</v>')
    os.makedirs(data_dir, exist_ok=True)
    json.dump(dataset, open(os.path.join(data_dir, 'data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument('--tasks', metavar='task_name', nargs='*', choices=['bio', 'dialogBIO', 'revert_dialogBIO'], help='names of tasks')
    parser.add_argument('--datasets', metavar='dataset_name', nargs='*', help='names of unified datasets')
    parser.add_argument('--save_dir', metavar='save_directory', type=str, default='data', help='directory to save the data, default: data/$task_name/$dataset_name')
    parser.add_argument('--infer_data_dir', metavar='infer_data_dir', type=str, default=None, help='directory of inference output data, default: None')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        dataset = load_dataset(dataset_name)
        for task_name in tqdm(args.tasks, desc='tasks', leave=False):
            data_dir = os.path.join(args.save_dir, task_name, dataset_name)
            eval(f"create_{task_name}_data")(dataset, data_dir, args)
