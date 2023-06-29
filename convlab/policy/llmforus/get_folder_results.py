from glob import glob
from argparse import ArgumentParser
import json
import numpy as np


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help="the folder path")
    return parser.parse_args()


def read_result(folder):
    result = {"Complete": [], "Success": [], "Success strict": []}
    for f_name in glob(folder + "/*.json"):
        with open(f_name, "r") as f:
            r = json.load(f)
            for x in result:
                result[x].append(r[x])
    for x, y in result.items():
        print(x, np.mean(y), np.std(y))


def main():
    args = arg_parse()
    read_result(args.folder)


if __name__ == "__main__":
    main()
