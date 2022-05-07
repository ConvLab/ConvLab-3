import json
import os
import random
from tqdm import tqdm

def main(args):
    random.seed(45)
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = [f for (_, _, fs) in os.walk(args.input_dir) for f in fs if 'keywords' in f]
    for filename in filenames:
        data = json.load(open(os.path.join(args.input_dir, filename)))
        fout = open(os.path.join(args.output_dir, f"{filename.split('/')[-1].split('_')[1]}.json"), 'w', encoding='utf-8')
        turn_keywords = [turn['keywords'] for dial in data for turn in dial]
        random.shuffle(turn_keywords)
        cnt = 0
        # keywords_set = {keyword for keywords in turn_keywords_set for keyword in keywords}
        for dial in tqdm(data):
            context = []
            for i, turn in enumerate(dial):
                speaker = 'user' if i%2 == 0 else 'system'
                random.shuffle(turn['keywords'])
                keywords = ' | '.join(turn['keywords'])
                utt = turn['utterance']
                context_seq = '\n'.join([f"{turn['speaker']}: {turn['utt']}" for turn in context]+[f'{speaker}: '])
                input_seq = f'keywords: {keywords}\n\ncontext: {context_seq}'
                context.append({'speaker': speaker, 'utt':utt})
                fout.write(json.dumps({'keywords+context': input_seq, 'response': utt}, ensure_ascii=False)+'\n')

                negative_keywords = turn_keywords[cnt]
                cnt += 1
                possible_keywords = turn['keywords'] + list(negative_keywords)
                random.shuffle(possible_keywords)
                possible_keywords = ' | '.join(possible_keywords)
                input_seq = f'possible keywords: {possible_keywords}\n\ncontext: {context_seq}'
                if args.noisy:
                    fout.write(json.dumps({'keywords+context': input_seq, 'response': utt}, ensure_ascii=False)+'\n')
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--input_dir', '-i', type=str, help='path to the input files')
    parser.add_argument('--output_dir', '-o', type=str, help='path to the output files')
    parser.add_argument('--noisy', action='store_true', help='whether add noisy keywords samples')
    args = parser.parse_args()
    print(args)
    main(args)
