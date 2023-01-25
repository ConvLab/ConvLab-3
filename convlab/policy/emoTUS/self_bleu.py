# from fast_bleu import SelfBLEU
import argparse
import json
from datasets import Dataset, load_metric
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    return parser.parse_args()


def read_file(file_name):
    nlg_candidates = json.load(open(file_name))
    return nlg_candidates


def get_sent(candidates):
    if "log" in candidates:
        return [x["gen_utts"] for x in candidates["log"]]
    else:
        return [x["gen_utts"] for x in candidates["dialog"]]

def SelfBLEU(sentences):
    metric = load_metric("sacrebleu")
    result = []
    for i, sent in tqdm(enumerate(sentences),ascii=True):
        r = metric.compute(predictions=[sent], references=[sentences[i:]+sentences[i+1:]])
        result.append(r["score"])


    return sum(result)/len(result)


def calculate(candidates):
    sentences = get_sent(candidates)
    bleu = SelfBLEU(sentences)
    x = bleu.get_score()
    print(x)


if __name__ == "__main__":
    args = arg_parser()
    calculate(read_file(args.file))
