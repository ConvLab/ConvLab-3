from argparse import ArgumentParser
from random import randint
from pprint import pprint
import json
import torch
from convlab.policy.emoUS.emoUS import UserActionPolicy
import os
from convlab.policy.emoUS.evaluate import emotion_score
from tqdm import tqdm


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--id", type=int, help="the sent id in test set.")
    parser.add_argument("--text", type=str, help="the conversation file")
    parser.add_argument("--test-file", type=str,
                        default="convlab/policy/emoUS/unify/data/language_emowoz+dialmage_0_1/test.json")
    parser.add_argument(
        "--model", type=str, default="convlab/policy/emoUS/unify/experiments/EmoUS_default/EmoUS_default")
    parser.add_argument("--whole", action="store_true",
                        help="evaluate the whole test set")
    return parser.parse_args()


def one_sent_evaluate(sent_id, text, test_data, model_checkpoint):
    if sent_id is None:
        emotion = "Neutral"
        while emotion == "Neutral":
            sent_id = randint(0, len(test_data))
            out_json = json.loads(test_data[sent_id]["out"])
            emotion = out_json["emotion"]
    in_json = json.loads(test_data[sent_id]["in"])
    if text is None:
        text = in_json["system"]
    print("Sent_ID", sent_id)
    print("----> goal")
    pprint(in_json["goal"])
    print("----> history")
    pprint(in_json["history"])
    print("----> system")
    print(text)
    model = load_model(model_checkpoint)
    emotion = generate(model, in_json, text)
    print("----> emotion")
    print(emotion)


def load_model(model_checkpoint):
    model = UserActionPolicy(model_checkpoint)
    model.load(os.path.join(
        model_checkpoint, "pytorch_model.bin"))
    return model


def generate(model: UserActionPolicy, in_json, text=None):
    if text is not None:
        in_json["system"] = text
    emotion = model.predict_emotion_from_text(in_json)

    return json.loads(emotion)


def whole_evaluate(test_data, model_checkpoint):
    model = load_model(model_checkpoint)
    preds = []
    labels = []
    result = {"dialog": []}
    for data in tqdm(test_data):
        in_json = json.loads(data["in"])
        out_json = json.loads(data["out"])
        pred = generate(model, in_json)
        preds.append(pred["emotion"])
        labels.append(out_json["emotion"])
        result["dialog"].append(
            {"input": data["in"], "golden_emotion": out_json["emotion"], "gen_emotion": pred["emotion"]})

    folder = os.path.join(model_checkpoint, "results")
    os.makedirs(folder, exist_ok=True)
    r = emotion_score(golden_emotions=labels,
                      gen_emotions=preds, dirname=folder)
    print(r)
    with open(os.path.join(model_checkpoint, "generation.json"), "w") as f:
        json.dump(result, f, indent=2)


def main():
    args = arg_parser()
    with open(args.test_file) as f:
        test_data = json.load(f)
    if args.whole:
        whole_evaluate(test_data["dialog"], args.model)
    else:
        one_sent_evaluate(args.id, args.text, test_data["dialog"], args.model)


if __name__ == "__main__":
    main()
