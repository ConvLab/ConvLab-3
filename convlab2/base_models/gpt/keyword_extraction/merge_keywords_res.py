import json

def main(args):
    filename2data = {f.split('/')[-1]: json.load(open(f)) for f in args.keywords_files}
    first_filename = args.keywords_files[0].split('/')[-1]
    dialogs = []
    for i in range(len(filename2data[first_filename])):
        turns = []
        for j in range(min([len(filename2data[filename][i]) for filename in filename2data])):
            utt = filename2data[first_filename][i][j]['utterance']
            keywords = {filename.split('_')[3]+'_nonstopword'+filename.split('_')[-1]: ' | '.join(filename2data[filename][i][j]['keywords']) for filename in filename2data}
            turns.append({
                "utterance": utt,
                **keywords
            })
        dialogs.append(turns)
    json.dump(dialogs, open(args.output_file, "w", encoding='utf-8'), indent=2, ensure_ascii=False)


    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--keywords_files', '-f', metavar='keywords_files', nargs='*', help='keywords files')
    parser.add_argument('--output_file', '-o', type=str, help='path to the output file')
    args = parser.parse_args()
    print(args)
    main(args)
