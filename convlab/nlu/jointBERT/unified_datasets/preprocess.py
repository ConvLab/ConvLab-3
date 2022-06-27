import json
import os
from collections import Counter
from convlab.util import load_dataset, load_nlu_data
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from tqdm import tqdm


def preprocess(dataset_name, speaker, save_dir, context_window_size):
    dataset = load_dataset(dataset_name)
    data_by_split = load_nlu_data(dataset, speaker=speaker, use_context=context_window_size>0, context_window_size=context_window_size)
    data_dir = os.path.join(save_dir, dataset_name, speaker, f'context_window_size_{context_window_size}')
    os.makedirs(data_dir, exist_ok=True)

    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()
    
    processed_data = {}
    all_tags = set([str(('O',))])
    all_intents = Counter()
    for data_split, data in data_by_split.items():
        if data_split == 'validation':
            data_split = 'val'
        processed_data[data_split] = []
        for sample in tqdm(data, desc=f'{data_split} samples'):

            utterance = sample['utterance']

            sentences = sent_tokenizer.tokenize(utterance)
            sent_spans = sent_tokenizer.span_tokenize(utterance)
            tokens = [token for sent in sentences for token in word_tokenizer.tokenize(sent)]
            token_spans = [(sent_span[0]+token_span[0], sent_span[0]+token_span[1]) for sent, sent_span in zip(sentences, sent_spans) for token_span in word_tokenizer.span_tokenize(sent)]
            tags = [str(('O',))] * len(tokens)
            for da in sample['dialogue_acts']['non-categorical']:
                if 'start' not in da:
                    # skip da that doesn't have span annotation
                    continue
                char_start = da['start']
                char_end = da['end']
                word_start, word_end = -1, -1
                for i, token_span in enumerate(token_spans):
                    if char_start == token_span[0]:
                        word_start = i
                    if char_end == token_span[1]:
                        word_end = i + 1
                if word_start == -1 and word_end == -1:
                    # char span does not match word, maybe there is an error in the annotation, skip
                    print('char span does not match word, skipping')
                    print('\t', 'utteance:', utterance)
                    print('\t', 'value:', utterance[char_start: char_end])
                    print('\t', 'da:', da, '\n')
                    continue
                intent, domain, slot = da['intent'], da['domain'], da['slot']
                all_tags.add(str((intent, domain, slot, 'B')))
                all_tags.add(str((intent, domain, slot, 'I')))
                tags[word_start] = str((intent, domain, slot, 'B'))
                for i in range(word_start+1, word_end):
                    tags[i] = str((intent, domain, slot, 'I'))

            intents = []
            for da in sample['dialogue_acts']['categorical']:
                intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da['value'].strip().lower()
                intent = str((intent, domain, slot, value))
                intents.append(intent)
                all_intents[intent] += 1
            for da in sample['dialogue_acts']['binary']:
                intent, domain, slot = da['intent'], da['domain'], da['slot']
                intent = str((intent, domain, slot))
                intents.append(intent)
                all_intents[intent] += 1
            context = []
            if context_window_size > 0:
                context = [s['utterance'] for s in sample['context']]
            processed_data[data_split].append([tokens, tags, intents, sample['dialogue_acts'], context])
        json.dump(processed_data[data_split], open(os.path.join(data_dir, '{}_data.json'.format(data_split)), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # filter out intents that occur only once to get intent vocabulary. however, these intents are still in the data
    all_intents = {x: count for x, count in all_intents.items() if count > 1}
    print('sentence label num:', len(all_intents))
    print('tag num:', len(all_tags))
    json.dump(sorted(all_intents), open(os.path.join(data_dir, 'intent_vocab.json'), 'w'), indent=2)
    json.dump(sorted(all_tags), open(os.path.join(data_dir, 'tag_vocab.json'), 'w'), indent=2)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create nlu data for bertnlu training")
    parser.add_argument('--dataset', '-d', metavar='dataset_name', type=str, help='name of the unified dataset')
    parser.add_argument('--speaker', '-s', type=str, choices=['user', 'system', 'all'], help='speaker(s) of utterances')
    parser.add_argument('--save_dir', metavar='save_directory', type=str, default='data', help='directory to save the data, save_dir/$dataset_name/$speaker')
    parser.add_argument('--context_window_size', '-c', type=int, default=0, help='how many contextual utterances are considered')
    args = parser.parse_args()
    print(args)
    preprocess(args.dataset, args.speaker, args.save_dir, args.context_window_size)
