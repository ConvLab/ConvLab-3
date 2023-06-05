from argparse import ArgumentParser
from random import randint
from pprint import pprint
import json
import torch
from convlab.policy.emoUS.emoUS import UserActionPolicy
import os


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--id", type=int, help="the sent id in test set.")
    parser.add_argument("--text", type=str, help="the conversation file")
    parser.add_argument("--test-file", type=str,
                        default="convlab/policy/emoUS/unify/data/language_emowoz+dialmage_0_1/test.json")
    parser.add_argument(
        "--model", type=str, default="convlab/policy/emoUS/unify/experiments/EmoUS_default/EmoUS_default")
    return parser.parse_args()


def evaluate(sent_id, text, test_data, model_checkpoint):
    if sent_id is None:
        sent_id = randint(0, len(test_data))
    in_json = json.loads(test_data[sent_id]["in"])
    if text is None:
        text = in_json["system"]
    print("----> goal")
    pprint(in_json["goal"])
    print("----> history")
    pprint(in_json["history"])
    print("----> system")
    print(text)
    model = load_model(model_checkpoint)
    generate(model, in_json, text)


def load_model(model_checkpoint):
    model = UserActionPolicy(model_checkpoint)
    model.load(os.path.join(
        model_checkpoint, "pytorch_model.bin"))
    return model


def generate(model: UserActionPolicy, in_json, text=None):
    if text is not None:
        in_json["system"] = text
    emotion = model.predict_emotion_from_text(in_json)
    print(emotion)


def main():
    args = arg_parser()
    with open(args.test_file) as f:
        test_data = json.load(f)
    evaluate(args.id, args.text, test_data["dialog"], args.model)


if __name__ == "__main__":
    main()
