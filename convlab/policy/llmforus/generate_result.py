import json
import os
from argparse import ArgumentParser

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--model-checkpoint", type=str, help="LLM checkpoint")
    parser.add_argument("--peft-checkpoint", type=str, help="PEFT checkpoint")
    return parser.parse_args()


def get_model(llm_checkpoint, peft_checkpoint):
    model = AutoModelForCausalLM.from_pretrained(
        llm_checkpoint, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, peft_checkpoint)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


def parse_emotion(text: str):
    text = text.split("the emotion of the user is:")[-1]
    text = text.strip()
    text = text.split(" ")[0]
    text = text.split('\n')[0]
    return text.strip()


def parse_utterance(text: str):
    text = text.split("your utterance is:")[-1]
    text = text.strip()
    return text.strip()


def evaluation(data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_token=30, output_dir="."):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    # stopping_criteria = StoppingCriteriaList(
    #     [StoppingCriteriaSub(stops='</s>')])
    emotion_results = []
    utterance_results = []
    for x in tqdm(data):
        if "emotion" in x["id"]:
            # only evaluate emotion
            inputs = tokenizer(x["in"], return_tensors="pt").to(device)
            generate_ids = model.generate(
                input_ids=inputs.input_ids, max_new_tokens=max_token,)
            output = parse_emotion(tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True)[0])
            emotion_results.append({"id": x["id"],
                                    "in": x["in"],
                                    "predict": output,
                                    "label": x["out"]})
        if "utterance" in x["id"]:
            # only evaluate emotion
            inputs = tokenizer(x["in"], return_tensors="pt").to(device)
            generate_ids = model.generate(
                input_ids=inputs.input_ids, max_new_tokens=max_token,)
            output = parse_utterance(tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True)[0])
            utterance_results.append({"id": x["id"],
                                      "in": x["in"],
                                      "predict": output,
                                      "label": x["out"]})
        # TODO how to evaluate action?

    with open(os.path.join(output_dir, "emotion_result.json"), "w") as fout:
        json.dump(emotion_results, fout, indent=2)
    with open(os.path.join(output_dir, "utterance_result.json"), "w") as fout:
        json.dump(utterance_results, fout, indent=2)


def main():
    args = arg_parser()
    model = get_model(args.model_checkpoint, args.peft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    data = json.load(open(args.data))
    result_folder = os.path.join(args.peft_checkpoint, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    evaluation(data['dialog'], model, tokenizer, output_dir=result_folder)


if __name__ == "__main__":
    main()
