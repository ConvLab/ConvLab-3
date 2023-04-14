from argparse import ArgumentParser

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="model name")
    parser.add_argument("--data", type=str)
    parser.add_argument("--gen-file", type=str)
    return parser.parse_args()


def main():
    args = arg_parser()
    model_checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint)
    input_text = "Yeah, I think we are. This isn't even my dress."
    inputs = tokenizer([input_text], return_tensors="pt", padding=True)
    output = model(input_ids=inputs["input_ids"],
                   attention_mask=inputs["attention_mask"])
    print(np.argmax(output, axis=-1))


if __name__ == "__main__":
    main()
