import json
import os

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = [f for (_, _, fs) in os.walk(args.input_dir) for f in fs if 'keywords' in f]
    for filename in filenames:
        data = json.load(open(os.path.join(args.input_dir, filename)))
        fout = open(os.path.join(args.output_dir, f"{filename.split('/')[-1].split('_')[1]}.json"), 'w', encoding='utf-8')
        for dial in data:
            context = []
            for i, turn in enumerate(dial):
                speaker = 'user' if i%2 == 0 else 'system'
                keywords = ', '.join(turn['keywords'])
                utt = turn['utterance']
                input_seq = '\n'.join([f"{turn['speaker']}: {turn['utt']}" for turn in context]+[f'{speaker}: '])
                input_seq = f'{keywords}\n{input_seq}'
                context.append({'speaker': speaker, 'utt':utt})
                fout.write(json.dumps({'keywords+context': input_seq, 'response': utt}, ensure_ascii=False)+'\n')
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--input_dir', '-i', type=str, help='path to the input files')
    parser.add_argument('--output_dir', '-o', type=str, help='path to the output files')
    args = parser.parse_args()
    print(args)
    main(args)
