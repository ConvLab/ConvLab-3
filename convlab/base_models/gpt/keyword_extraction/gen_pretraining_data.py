import json
import os
import random
from tqdm import tqdm

def main(args):
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = [f for (_, _, fs) in os.walk(args.input_dir) for f in fs if 'keywords' in f]
    for filename in filenames:
        data = json.load(open(os.path.join(args.input_dir, filename)))
        fout = open(os.path.join(args.output_dir, f"{filename.split('/')[-1].split('_')[1]}.json"), 'w', encoding='utf-8')
        for dial in tqdm(data):
            context = []
            turns_keywords = [turn['keywords'] for turn in dial]
            for i, turn in enumerate(dial):
                speaker = 'user' if i % 2 == 0 else 'system'
                utt = turn['utterance']
                context_seq = '\n'.join([f"{turn['speaker']}: {turn['utt']}" for turn in context]+[f'{speaker}: '])
                context.append({'speaker': speaker, 'utt': utt})
                if i == 0:
                    continue
                
                input_seq = f'generate a response: context:\n\n{context_seq}'
                fout.write(json.dumps({'source': input_seq, 'target': utt}, ensure_ascii=False)+'\n')
                if args.mode == 'rg':
                    continue

                random.shuffle(turn['keywords'])
                for j in range(len(turn['keywords'])):
                    random.shuffle(turn['keywords'][j])
                keywords = ' | '.join([' : '.join(sent_keywords) for sent_keywords in turn['keywords']])
                input_seq = f'generate a response: grounded knowledge: | {keywords} | context:\n\n{context_seq}'
                fout.write(json.dumps({'source': input_seq, 'target': utt, 'keywords': turn['keywords']}, ensure_ascii=False)+'\n')
                if args.mode == 'key2gen':
                    continue

                possible_keywords_sents = turn['keywords'][:]
                num_possible_keywords_turns = min(random.randint(1, 5), len(turns_keywords) - 1)
                for turn_keywords in random.sample(turns_keywords[:i] + turns_keywords[i+1:], num_possible_keywords_turns):
                    possible_keywords_sents.extend(turn_keywords)
                random.shuffle(possible_keywords_sents)
                possible_keywords = ' | '.join([' : '.join(sent_keywords) for sent_keywords in possible_keywords_sents])
                input_seq = f'generate a response: all knowledge: | {possible_keywords} | context:\n\n{context_seq}'
                fout.write(json.dumps({'source': input_seq, 'target': utt, 'keywords': turn['keywords'], 'all_keywords': possible_keywords_sents}, ensure_ascii=False)+'\n')
                if args.mode == 'key2gen_noisy':
                    continue
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--input_dir', '-i', type=str, help='path to the input files')
    parser.add_argument('--output_dir', '-o', type=str, help='path to the output files')
    parser.add_argument('--mode', '-m', type=str, choices=['rg', 'key2gen', 'key2gen_noisy'], help='which task to perform')
    args = parser.parse_args()
    print(args)
    main(args)
