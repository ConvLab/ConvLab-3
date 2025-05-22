import json
import os
from argparse import ArgumentParser

import pandas as pd


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--test", type=str, default="")
    parser.add_argument("--augment", type=str, default="")
    parser.add_argument("--out-file", '-o', type=str, default="augment.json")
    return parser.parse_args()


def json2dict(f):
    d = {}
    for x in f["dialog"]:
        d[x['id']] = x
    return d


def main():
    args = arg_parser()
    t = json2dict(json.load(open(args.test)))
    a = pd.read_csv(args.augment)
    r = {"dialog": []}
    for _, l in a.iterrows():
        if "DMAGE" not in l["dialogue_id"]:
            continue
        data_id = f'{l["dialogue_id"]}-{l["turn_id"]+1}'
        temp = {}
        if data_id in t:

            s = l["generated_utterace"]
            temp["id"] = f'{data_id}-{l["target_conduct"]}'
            temp["in"] = json.loads(t[data_id]["in"])
            temp["in"]["system"] = s
            temp["in"] = json.dumps(temp["in"])
            temp["act"] = t[data_id]["act"]
            temp["out"] = t[data_id]["out"]
            r["dialog"].append(temp)
    json.dump(r, open(args.out_file, "w"), indent=2)


if __name__ == "__main__":
    main()
