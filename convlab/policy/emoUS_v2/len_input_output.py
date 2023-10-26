from argparse import ArgumentParser
import json
from transformers import AutoTokenizer


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", "-d", type=str, help="data path")
    parser.add_argument("--model-checkpoint", "-m",
                        type=str, help="model checkpoint")
    return parser.parse_args()


def calculate(data, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    input_len = 0
    output_len = 0
    for d in data["dialog"]:
        x = len(tokenizer(d["in"])[0])
        if x > input_len:
            input_len = x

        x = len(tokenizer(d["out"])[0])
        if x > output_len:
            output_len = x
    print("input len:", input_len)
    print("output len:", output_len)


if __name__ == "__main__":
    args = arg_parser()
    data = json.load(open(args.data))
    calculate(data, args.model_checkpoint)
