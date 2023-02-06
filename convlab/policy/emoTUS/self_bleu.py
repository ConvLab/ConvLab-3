# from fast_bleu import SelfBLEU
import argparse
import json
from datasets import Dataset, load_metric
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--fast-bleu", action="store_true")
    parser.add_argument("--uss", action="store_true")
    return parser.parse_args()


def read_file(file_name):
    nlg_candidates = json.load(open(file_name))
    return nlg_candidates


def get_sent(candidates, bleu_mode="torch", uss=False):
    if bleu_mode == "torch":
        if uss:
            return [x["preds"] for x in candidates]
        if "log" in candidates:
            return [x["gen_utts"] for x in candidates["log"]]
        else:
            return [x["gen_utts"] for x in candidates["dialog"]]
    else:
        if uss:
            return [x["preds"].split() for x in candidates]
        if "log" in candidates:
            return [x["gen_utts"].split() for x in candidates["log"]]
        else:
            return [x["gen_utts"].split() for x in candidates["dialog"]]


def SelfBLEU(sentences):
    metric = load_metric("sacrebleu")
    result = []
    for i, sent in tqdm(enumerate(sentences), ascii=True):
        r = metric.compute(predictions=[sent], references=[
                           sentences[i:]+sentences[i+1:]])
        result.append(r["score"])

    return sum(result)/len(result)


def calculate(candidates, bleu_mode="torch", uss=False):
    sentences = get_sent(candidates, bleu_mode, uss)
    if bleu_mode == "torch":
        x = SelfBLEU(sentences)
    else:
        bleu = fast_bleu.SelfBLEU(sentences)
        x = bleu.get_score()
    # x = bleu.get_score()
    # print(x)
    print(sum(x[4])/len(x[4]))


if __name__ == "__main__":
    args = arg_parser()
    if args.fast_bleu:
        import fast_bleu
        calculate(read_file(args.file), "fast-bleu", uss=args.uss)
    else:
        calculate(read_file(args.file), uss=args.uss)
