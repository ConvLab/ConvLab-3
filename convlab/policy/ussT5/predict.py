from argparse import ArgumentParser

from transformers import T5ForConditionalGeneration, T5Tokenizer


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="model name")

    return parser.parse_args()


def main():
    args = arg_parser()
    model_checkpoint = args.model
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    prefix = 'satisfaction score: '
    text = "hi, i'm looking for an attraction in the center of town to visit. we have quite a few interesting attractions in the center of town. is there anything in particular you would like to see?"
    inputs = tokenizer([prefix+text], return_tensors="pt", padding=True)
    output = model.generate(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            do_sample=False)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))

if __name__ == "__main__":
    main()
