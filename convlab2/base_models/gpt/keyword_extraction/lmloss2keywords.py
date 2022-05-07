import json
import json_lines
from pprint import pprint
import os
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import GPT2Tokenizer
from string import punctuation


def merge_tokens(tokens, losses):
    """Merge tokens into words"""
    res = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        loss = losses[i]
        if token in ['Ġ', 'Ċ']:
            # "Ġ" means " ", "Ċ" means "\n"
            if token == 'Ċ' and i < len(tokens) - 1 and not tokens[i+1].startswith('Ġ'):
                tokens[i+1] = 'Ġ'+tokens[i+1]
            i += 1
            continue
        if token in ['user', 'system', 'Ġuser', 'Ġsystem'] and i < len(tokens)-1 and tokens[i+1] == ':':
            if i > 0:
                tokens[i+1] = '<|endoftext|>'
                i += 1
            else:
                i += 2
            continue
        if token.startswith('Ġ'):
            # token = token.replace("Ġ", "")
            res.append([[token], [loss]])
        elif token == '<|endoftext|>':
            res.append([[token], [loss]])
        else:
            assert 'Ġ' not in token
            if len(res) > 0:
                res[-1][0].append(token)
                res[-1][1].append(loss)
            else:
                res.append([token, loss])
        i += 1
    return res


def convert_token_loss2word_loss(token_loss_file):
    """generate a word loss file according to the token loss file"""
    word_loss_file = os.path.join(os.path.dirname(token_loss_file), token_loss_file.split('/')[-1].replace('token', 'word'))
    fin = open(token_loss_file, 'rb')
    fout = open(word_loss_file, 'w', encoding='utf-8')
    lines = []

    for item in tqdm(json_lines.reader(fin)):
        tokens, losses = item['tokens'], item['losses']
        assert len(tokens) == len(losses)
        word2losses = merge_tokens(tokens, losses)
        lines.append({"words": [x[0] for x in word2losses], "losses": [x[1] for x in word2losses]})
        fout.write(json.dumps(lines[-1], ensure_ascii=False)+'\n')

    fin.close()
    fout.close()
    return lines

def main(args):
    if not args.word_loss_file:
        word_loss_list = convert_token_loss2word_loss(args.token_loss_file)
    else:
        fin = open(args.word_loss_file, 'rb')
        word_loss_list = []
        for item in json_lines.reader(fin):
            words, losses = item['words'], item['losses']
            word_loss_list.append({"words": words, "losses": losses})
        fin.close()

    if not args.output_file:
        return

    stop_words = set(stopwords.words('english'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    if args.keywords_th_ratio > 0:
        losses = [loss for x in word_loss_list for word, loss in zip(x['words'], x['losses']) if not any([w.lower() in stop_words for w in word_tokenize(word)])]
        loss_th = sorted(losses, reverse=True)[round(args.keywords_th_ratio*len(losses))]
        print(f'loss th for top {args.keywords_th_ratio*100}%: {loss_th}')
    else:
        loss_th = 0

    def keywords_filter(word_loss_pairs):
        index2keyword = {}
        for i, word_loss_pair in enumerate(word_loss_pairs):
            words = word_tokenize(word_loss_pair[0])
            if args.stopwords and any([w.lower() in stop_words for w in words]):
                # skip stopwords
                continue
            if word_loss_pair[1] <= loss_th:
                # skip if loss is too small
                continue
            # strip punctuation
            strip_punctuation = word_loss_pair[0].strip(punctuation).strip()
            if len(strip_punctuation) == 0:
                # skip punctuation
                continue
            index2keyword[i] = strip_punctuation
        candidate_indexes = list(index2keyword.keys())
        topk = min(round(args.keywords_ratio*len(word_loss_pairs)), args.keywords_num)
        topk_indexes = sorted(candidate_indexes, key=lambda x: word_loss_pairs[x][1], reverse=True)[:topk]
        topk_indexes = sorted(topk_indexes)
        keywords = []
        for i, index in enumerate(topk_indexes):
            if i > 0 and index == topk_indexes[i-1] + 1 and \
                word_loss_pairs[index][0].strip().startswith(index2keyword[index]) and \
                word_loss_pairs[topk_indexes[i-1]][0].strip().endswith(index2keyword[topk_indexes[i-1]]):
                keywords[-1]+= ' '+index2keyword[index]
            else:
                keywords.append(index2keyword[index])

        return keywords

    dialogs = []
    for item in tqdm(word_loss_list):
        words = item['words']
        losses = item['losses']
        turns = []
        turn = {'words': [], 'losses': []}
        for word, loss in zip(words, losses):
            if word == ['<|endoftext|>']:
                # switch turn
                turn['words'] = [tokenizer.convert_tokens_to_string(tokens) for tokens in turn['words']]
                turn['losses'] = [np.mean(losses) for losses in turn['losses']]
                turn['utterance'] = ''.join(turn['words']).strip()
                keywords = keywords_filter(list(zip(turn['words'], turn['losses'])))
                turn['keywords'] = keywords
                turn.pop('words')
                turn.pop('losses')
                turns.append(turn)
                turn = {'words': [], 'losses': []}
            else:
                turn['words'].append(word)
                turn['losses'].append(loss)
        dialogs.append(turns)
    json.dump(dialogs, open(args.output_file, "w", encoding='utf-8'), indent=2, ensure_ascii=False)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="extract keywords according to lm loss")
    parser.add_argument('--model_type', '-m', type=str, help='gpt or dialogpt')
    parser.add_argument('--token_loss_file', '-t', type=str, help='path to the token loss file that contains two columns: [tokens, losses]')
    parser.add_argument('--word_loss_file', '-w', type=str, help='path to the token loss file that contains two columns: [tokens, losses]')
    parser.add_argument('--output_file', '-o', type=str, help='path to the output file')
    parser.add_argument('--keywords_num', '-n', type=int, default=100, help='how many words in an utterance serve as keywords')
    parser.add_argument('--keywords_ratio', '-r', type=float, default=1.0, help='how many words (in ratio) in an utterance serve as keywords')
    parser.add_argument('--keywords_th_ratio', '-th', type=float, default=0., help='loss threshold for the keywords, ratio of all word losses')
    parser.add_argument('--stopwords', '-s', type=lambda x: bool(eval(x)), default=True, help='filter out stopwords')
    args = parser.parse_args()
    print(args)
    main(args)
