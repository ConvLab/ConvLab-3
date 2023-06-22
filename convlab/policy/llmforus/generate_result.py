import json
import os
from argparse import ArgumentParser

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from convlab.policy.llmforus.knowledge_graph import KnowledgeGraph
import re
from convlab.policy.genTUS.stepGenTUSmodel import stepGenTUSmodel
from convlab.policy.llmforus.token_map import tokenMap
import time


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, help="data path")
    parser.add_argument("--model-checkpoint", type=str, help="LLM checkpoint")
    parser.add_argument("--peft-checkpoint", type=str, help="PEFT checkpoint")
    return parser.parse_args()


def get_model(llm_checkpoint, peft_checkpoint):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = stepGenTUSmodel(
        llm_checkpoint,
        device=device,
        model_type="LLM",
        peft_model_checkpoint=peft_checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(
    #     llm_checkpoint, torch_dtype=torch.float16)
    # model = PeftModel.from_pretrained(model, peft_checkpoint)

    # model.to(device)
    # model.eval()
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


def parse_direct_action(text: str):
    text = text.split("your action is:")[-1]
    text = text.strip()
    return text.strip()


def parse_input_string(input_string):
    pattern = r"The (\w+) (\w+)-(\w+): (.+) is (\w+)."
    match = re.match(pattern, input_string)

    if match:
        intent = match.group(1)
        domain = match.group(2)
        slot = match.group(3)
        value = match.group(4)
        # status = match.group(5)

        return [intent, domain, slot, value]

    return None


def get_goal(text: str):
    text = text.split("The given goal status is:\n")[-1]
    text = text.split("You feel")[0]
    goal = []
    for line in text.split("\n"):
        x = parse_input_string(line)
        if x:
            goal.append(x)
    return goal


class SemanticActionGenerator:
    def __init__(self, model, tokenizer):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        model_type = "LLM"
        self.model = model
        self.tokenizer = tokenizer
        self.kg = KnowledgeGraph(
            tokenizer=tokenizer,
            model_type=model_type)
        self.max_out_len = 100
        self.token_map = tokenMap(tokenizer=tokenizer, model_type=model_type,)
        self.max_action_len = 3

    def generate(self, input_text, mode="max", allow_general_intent=True, max_act_len=3):
        self.max_action_len = max_act_len
        goal = get_goal(input_text)
        self.kg.init_from_given_goal(goal)
        self.mentioned_domain = []  # TODO need to be updateds

        model_input = self.tokenizer(
            input_text, return_tensors="pt").to(self.device)

        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq(self.token_map.get_id('start_json'), 0)

        for act_len in range(max_act_len):
            pos = self._get_act(
                model_input, pos, mode, allow_general_intent)

            terminate, token_name = self._stop_semantic(
                model_input, pos, act_len)
            pos = self._update_seq(self.token_map.get_id(token_name), pos)

            if terminate:
                break
        action = self.tokenizer.batch_decode(
            self.seq, skip_special_tokens=True)[0]
        return action

    def _stop_semantic(self, model_input, pos, act_length=0):

        outputs = self.model.get_next_token_logits(
            model_input, self.seq[:1, :pos])
        tokens = {}
        for token_name in ['sep_act', 'end_act']:
            tokens[token_name] = {
                "token_id": self.token_map.get_id(token_name)}
            hash_id = tokens[token_name]["token_id"][0]
            tokens[token_name]["score"] = outputs[:, hash_id].item()

        if tokens['end_act']["score"] > tokens['sep_act']["score"]:
            terminate = True
        else:
            terminate = False

        if act_length >= self.max_action_len - 1:
            terminate = True

        token_name = "end_act" if terminate else "sep_act"

        return terminate, token_name

    def _get_act(self, model_input, pos, mode="max", allow_general_intent=True):
        intent = self._get_intent(
            model_input, self.seq[:1, :pos], mode, allow_general_intent)
        pos = self._update_seq(intent["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get domain
        domain = self._get_domain(
            model_input, self.seq[:1, :pos], intent["token_name"], mode)
        pos = self._update_seq(domain["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get slot
        slot = self._get_slot(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], mode)
        if "book" in slot["token_name"]:
            pos = self._update_seq(self.token_map.get_id('book'), pos)
            slot = self._get_book_slot(
                model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], mode)
            slot["token_name"] = "book" + slot["token_name"]
        pos = self._update_seq(slot["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get value

        value = self._get_value(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], slot["token_name"], mode)
        pos = self._update_seq(value["token_id"], pos)

        act = [intent["token_name"], domain["token_name"],
               slot["token_name"], value["token_name"]]
        return pos

    def _update_seq(self, sub_seq: list, pos: int):
        for x in sub_seq:
            self.seq[0, pos] = x
            pos += 1

        return pos

    def _get_intent(self, model_input, generated_so_far, mode="max", allow_general_intent=True):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_intent(next_token_logits, mode, allow_general_intent)

    def _get_domain(self, model_input, generated_so_far, intent, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_domain(next_token_logits, intent, mode)

    def _get_slot(self, model_input, generated_so_far, intent, domain, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        is_mentioned = self.is_mentioned(domain)
        return self.kg.get_slot(next_token_logits, intent, domain, mode, is_mentioned)

    def _get_book_slot(self, model_input, generated_so_far, intent, domain, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        is_mentioned = self.is_mentioned(domain)
        return self.kg.get_book_slot(next_token_logits, intent, domain, mode, is_mentioned)

    def _get_value(self, model_input, generated_so_far, intent, domain, slot, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_value(next_token_logits, intent, domain, slot, mode)

    def is_mentioned(self, domain):
        if domain in self.mentioned_domain:
            return True
        return False


def get_action(text: str, generator: SemanticActionGenerator, max_act_len=2):
    act = generator.generate(text, max_act_len=max_act_len)
    return act


def get_emotion(text, model, tokenizer, device, max_token):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generate_ids = model.generate(
        input_ids=inputs.input_ids, max_new_tokens=max_token,)
    output = parse_emotion(tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True)[0])
    return output


def direct_get_action(text, model, tokenizer, device, max_token):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generate_ids = model.generate(
        input_ids=inputs.input_ids, max_new_tokens=max_token,)
    output = parse_direct_action(tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True)[0])
    return output


def get_utterance(text, model, tokenizer, device, max_token):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generate_ids = model.generate(
        input_ids=inputs.input_ids, max_new_tokens=max_token,)
    output = parse_utterance(tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True)[0])
    return output


def evaluation(data: dict, model: stepGenTUSmodel, tokenizer: AutoTokenizer, max_token=30, output_dir="."):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    # stopping_criteria = StoppingCriteriaList(
    #     [StoppingCriteriaSub(stops='</s>')])
    emotion_results = []
    utterance_results = []
    action_results = []
    act_generator = SemanticActionGenerator(model, tokenizer)
    force_time = []
    direct_time = []

    for x in tqdm(data):
        if "emotion" in x["id"]:
            pass
            # only evaluate emotion
            output = get_emotion(
                x["in"], model.model, tokenizer, device, max_token)
            emotion_results.append({"id": x["id"],
                                    "in": x["in"],
                                    "predict": output,
                                    "label": x["out"]})

        if "utterance" in x["id"]:
            pass
            # only evaluate emotion
            output = get_utterance(
                x["in"],  model.model, tokenizer, device, max_token)
            utterance_results.append({"id": x["id"],
                                      "in": x["in"],
                                      "predict": output,
                                      "label": x["out"]})
        if "action" in x["id"]:
            t1 = time.time()
            output = get_action(x["in"], act_generator, max_act_len=2)
            t2 = time.time()
            direct_output = direct_get_action(
                x["in"],  model.model, tokenizer, device, max_token=100)
            t3 = time.time()
            force_time.append(t2-t1)
            direct_time.append(t3-t2)

            action_results.append({"id": x["id"],
                                   "in": x["in"],
                                   "predict": output,
                                   "direct_predict": direct_output,
                                   "label": x["out"]})
        # TODO how to evaluate action?
    print("force time: ", sum(force_time)/len(force_time))
    print("direct time: ", sum(direct_time)/len(direct_time))

    with open(os.path.join(output_dir, "emotion_result.json"), "w") as fout:
        json.dump(emotion_results, fout, indent=2)
    with open(os.path.join(output_dir, "utterance_result.json"), "w") as fout:
        json.dump(utterance_results, fout, indent=2)
    with open(os.path.join(output_dir, "action_result.json"), "w") as fout:
        json.dump(action_results, fout, indent=2)
    with open(os.path.join(output_dir, "time.json"), "w") as fout:
        json.dump({"force_time": force_time,
                  "direct_time": direct_time}, fout, indent=2)


def main():
    args = arg_parser()
    model = get_model(args.model_checkpoint, args.peft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    data = json.load(open(args.data))
    result_folder = os.path.join(args.peft_checkpoint, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    t = time.time()
    evaluation(data['dialog'], model, tokenizer, output_dir=result_folder)
    print("evaluation time: ", time.time() - t)


if __name__ == "__main__":
    main()
