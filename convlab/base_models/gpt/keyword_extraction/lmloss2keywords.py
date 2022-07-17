import json
import json_lines
from pprint import pprint
import os
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
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
            res.append([[token], [0.]])
        else:
            assert 'Ġ' not in token
            if len(res) > 0:
                res[-1][0].append(token)
                res[-1][1].append(loss)
            else:
                res.append([[token], [loss]])
        i += 1
    return res


def convert_token_loss2word_loss(token_loss_file):
    """generate a word loss file according to the token loss file"""
    word_loss_file = os.path.join(os.path.dirname(token_loss_file), token_loss_file.split('/')[-1].replace('token', 'word'))
    fin = open(token_loss_file, 'rb')
    fout = open(word_loss_file, 'w', encoding='utf-8')

    for item in tqdm(json_lines.reader(fin)):
        tokens, losses = item['tokens'], item['losses']
        assert len(tokens) == len(losses)
        word2losses = merge_tokens(tokens, losses)
        fout.write(json.dumps({"words": [x[0] for x in word2losses], "losses": [x[1] for x in word2losses]}, ensure_ascii=False)+'\n')

    fin.close()
    fout.close()
    return word_loss_file

def main(args):
    if not args.word_loss_file:
        word_loss_file = convert_token_loss2word_loss(args.token_loss_file)
    else:
        word_loss_file = args.word_loss_file

    if not args.output_file:
        return

    stop_words = set(stopwords.words('english'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    sent_tokenizer = PunktSentenceTokenizer()

    def keywords_filter(words, losses):
        word_loss_pairs = list(zip(words, losses))
        index2keyword = {}
        index2turn_sent = {}
        num_turns = 0
        turns_sent_spans = [list(sent_tokenizer.span_tokenize(utt)) for utt in ''.join(words).strip().split('<|endoftext|>')]
        utt = ''
        for i, word_loss_pair in enumerate(word_loss_pairs):
            if word_loss_pair[0].startswith('<|endoftext|>'):
                num_turns += 1
                utt = ''
                continue
            utt += word_loss_pair[0]
            words = word_tokenize(word_loss_pair[0])
            if args.stopwords and any([w.lower() in stop_words for w in words]):
                # skip stopwords
                continue
            if word_loss_pair[1] <= args.keywords_loss_th:
                # skip if loss is too small
                continue
            # strip punctuation
            strip_punctuation = word_loss_pair[0].strip(punctuation).strip()
            if len(strip_punctuation) == 0:
                # skip punctuation
                continue
            index2keyword[i] = strip_punctuation
            for sent_idx, (sent_start, sent_end) in enumerate(turns_sent_spans[num_turns]):
                if len(utt.strip()) <= sent_end:
                    index2turn_sent[i] = (num_turns, sent_idx)
                    break
        candidate_indexes = list(index2keyword.keys())
        topk = min(round(args.keywords_ratio*(len(word_loss_pairs)-num_turns)), args.keywords_num)
        topk_indexes = sorted(candidate_indexes, key=lambda x: word_loss_pairs[x][1], reverse=True)[:topk]
        topk_indexes = sorted(topk_indexes)
        keywords = []
        keywords_turn_sent2idx = {}
        for i, index in enumerate(topk_indexes):
            if i > 0 and index == topk_indexes[i-1] + 1 and \
                word_loss_pairs[index][0].strip().startswith(index2keyword[index]) and \
                word_loss_pairs[topk_indexes[i-1]][0].strip().endswith(index2keyword[topk_indexes[i-1]]):
                keywords[-1]+= ' '+index2keyword[index]
            else:
                keywords_turn_sent2idx.setdefault(index2turn_sent[index][0], {})
                keywords_turn_sent2idx[index2turn_sent[index][0]].setdefault(index2turn_sent[index][1], [])
                keywords_turn_sent2idx[index2turn_sent[index][0]][index2turn_sent[index][1]].append(len(keywords))
                keywords.append(index2keyword[index])

        return keywords, keywords_turn_sent2idx

    fin = open(word_loss_file, 'rb')
    fout = open(args.output_file, 'w', encoding='utf-8')

    for item in json_lines.reader(fin):
        words = [tokenizer.convert_tokens_to_string(tokens) for tokens in item['words']]
        losses = [np.mean(loss) for loss in item['losses']]
        dialog_keywords, keywords_turn_sent2idx = keywords_filter(words, losses)
        # print(keywords_turn_sent2idx)
        turns = []
        turn = {'words': [], 'losses': []}
        for i, (word, loss) in enumerate(zip(words, losses)):
            if word != '<|endoftext|>':
                turn['words'].append(word)
                turn['losses'].append(loss)
            if word == '<|endoftext|>' or i == len(words) - 1:
                # switch turn
                turn['utterance'] = ''.join(turn['words']).strip()
                # 1) extract keywords according to LM loss within the turn
                # keywords, _ = keywords_filter(turn['words'], turn['losses'])
                # turn['turn-level_keywords'] = keywords
                # 1) extract keywords according to LM loss over the dialog, and group them by sentence
                turn['keywords'] = [[dialog_keywords[idx] for idx in k_idxes] for sent_idx, k_idxes in keywords_turn_sent2idx.get(len(turns), {}).items()]
                turn.pop('words')
                turn.pop('losses')
                turns.append(turn)
                turn = {'words': [], 'losses': []}
                
        fout.write(json.dumps(turns, ensure_ascii=False)+'\n')
    
    fin.close()
    fout.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="extract keywords according to lm loss")
    parser.add_argument('--model_type', '-m', type=str, help='gpt or dialogpt')
    parser.add_argument('--model_name_or_path', type=str, help='model name or path')
    parser.add_argument('--token_loss_file', '-t', type=str, help='path to the token loss file that contains two columns: [tokens, losses]')
    parser.add_argument('--word_loss_file', '-w', type=str, help='path to the token loss file that contains two columns: [tokens, losses]')
    parser.add_argument('--output_file', '-o', type=str, help='path to the output file')
    parser.add_argument('--keywords_num', '-n', type=int, default=100, help='how many words in an utterance serve as keywords')
    parser.add_argument('--keywords_ratio', '-r', type=float, default=1.0, help='how many words (in ratio) in an utterance serve as keywords')
    parser.add_argument('--keywords_loss_th', '-th', type=float, default=0., help='loss threshold for the keywords')
    parser.add_argument('--stopwords', '-s', type=lambda x: bool(eval(x)), default=True, help='filter out stopwords')
    args = parser.parse_args()
    print(args)
    main(args)
