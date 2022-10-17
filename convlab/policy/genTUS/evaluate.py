import json
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import torch
from convlab.nlg.evaluate import fine_SER
from datasets import load_metric

# from convlab.policy.genTUS.pg.stepGenTUSagent import \
#     stepGenTUSPG as UserPolicy
from convlab.policy.genTUS.stepGenTUS import UserActionPolicy
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, help="the model path")
    parser.add_argument("--model-weight", type=str,
                        help="the model weight", default="")
    parser.add_argument("--input-file", type=str, help="the testing input file",
                        default="")
    parser.add_argument("--generated-file", type=str, help="the generated results",
                        default="")
    parser.add_argument("--only-action", action="store_true")
    parser.add_argument("--dataset", default="multiwoz")
    parser.add_argument("--do-semantic", action="store_true",
                        help="do semantic evaluation")
    parser.add_argument("--do-nlg", action="store_true",
                        help="do nlg generation")
    parser.add_argument("--do-golden-nlg", action="store_true",
                        help="do golden nlg generation")
    return parser.parse_args()


class Evaluator:
    def __init__(self, model_checkpoint, dataset, model_weight=None, only_action=False):
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.model_weight = model_weight
        # if model_weight:
        #     self.usr_policy = UserPolicy(
        #         self.model_checkpoint, only_action=only_action)
        #     self.usr_policy.load(model_weight)
        #     self.usr = self.usr_policy.usr
        # else:
        self.usr = UserActionPolicy(
            model_checkpoint, only_action=only_action, dataset=self.dataset)
        self.usr.load(os.path.join(model_checkpoint, "pytorch_model.bin"))

    def generate_results(self, f_eval, golden=False):
        in_file = json.load(open(f_eval))
        r = {
            "input": [],
            "golden_acts": [],
            "golden_utts": [],
            "gen_acts": [],
            "gen_utts": []
        }
        for dialog in tqdm(in_file['dialog']):
            inputs = dialog["in"]
            labels = self.usr._parse_output(dialog["out"])
            if golden:
                usr_act = labels["action"]
                usr_utt = self.usr.generate_text_from_give_semantic(
                    inputs, usr_act)

            else:
                output = self.usr._parse_output(
                    self.usr._generate_action(inputs))
                usr_act = self.usr._remove_illegal_action(output["action"])
                usr_utt = output["text"]
            r["input"].append(inputs)
            r["golden_acts"].append(labels["action"])
            r["golden_utts"].append(labels["text"])
            r["gen_acts"].append(usr_act)
            r["gen_utts"].append(usr_utt)

        return r

    def read_generated_result(self, f_eval):
        in_file = json.load(open(f_eval))
        r = {
            "input": [],
            "golden_acts": [],
            "golden_utts": [],
            "gen_acts": [],
            "gen_utts": []
        }
        for dialog in tqdm(in_file['dialog']):
            for x in dialog:
                r[x].append(dialog[x])

        return r

    def nlg_evaluation(self, input_file=None, generated_file=None, golden=False):
        if input_file:
            print("Force generation")
            gen_r = self.generate_results(input_file, golden)

        elif generated_file:
            gen_r = self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")

        nlg_eval = {
            "golden": golden,
            "metrics": {},
            "dialog": []
        }
        for input, golden_act, golden_utt, gen_act, gen_utt in zip(gen_r["input"], gen_r["golden_acts"], gen_r["golden_utts"], gen_r["gen_acts"], gen_r["gen_utts"]):
            nlg_eval["dialog"].append({
                "input": input,
                "golden_acts": golden_act,
                "golden_utts": golden_utt,
                "gen_acts": gen_act,
                "gen_utts": gen_utt
            })

        if golden:
            print("Calculate BLEU")
            bleu_metric = load_metric("sacrebleu")
            labels = [[utt] for utt in gen_r["golden_utts"]]

            bleu_score = bleu_metric.compute(predictions=gen_r["gen_utts"],
                                             references=labels,
                                             force=True)
            print("bleu_metric", bleu_score)
            nlg_eval["metrics"]["bleu"] = bleu_score

        else:
            print("Calculate SER")
            missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
                gen_r["gen_acts"], gen_r["gen_utts"])

            print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
                "genTUSNLG", missing, total, hallucinate, missing/total))
            nlg_eval["metrics"]["SER"] = missing/total

        dir_name = self.model_checkpoint
        json.dump(nlg_eval,
                  open(os.path.join(dir_name, "nlg_eval.json"), 'w'),
                  indent=2)
        return os.path.join(dir_name, "nlg_eval.json")

    def evaluation(self, input_file=None, generated_file=None):
        force_prediction = True
        if generated_file:
            gen_file = json.load(open(generated_file))
            force_prediction = False
            if gen_file["golden"]:
                force_prediction = True

        if force_prediction:
            in_file = json.load(open(input_file))
            dialog_result = []
            gen_acts, golden_acts = [], []
            # scores = {"precision": [], "recall": [], "f1": [], "turn_acc": []}
            for dialog in tqdm(in_file['dialog']):
                inputs = dialog["in"]
                labels = self.usr._parse_output(dialog["out"])
                ans_action = self.usr._remove_illegal_action(labels["action"])
                preds = self.usr._generate_action(inputs)
                preds = self.usr._parse_output(preds)
                usr_action = self.usr._remove_illegal_action(preds["action"])

                gen_acts.append(usr_action)
                golden_acts.append(ans_action)

                d = {"input": inputs,
                     "golden_acts": ans_action,
                     "gen_acts": usr_action}
                if "text" in preds:
                    d["golden_utts"] = labels["text"]
                    d["gen_utts"] = preds["text"]
                    # print("pred text", preds["text"])

                dialog_result.append(d)
        else:
            gen_acts, golden_acts = [], []
            for dialog in gen_file['dialog']:
                gen_acts.append(dialog["gen_acts"])
                golden_acts.append(dialog["golden_acts"])
            dialog_result = gen_file['dialog']

        scores = {"precision": [], "recall": [], "f1": [], "turn_acc": []}

        for gen_act, golden_act in zip(gen_acts, golden_acts):
            s = f1_measure(preds=gen_act, labels=golden_act)
            for metric in scores:
                scores[metric].append(s[metric])

        result = {}
        for metric in scores:
            result[metric] = sum(scores[metric])/len(scores[metric])
            print(f"{metric}: {result[metric]}")

        result["dialog"] = dialog_result
        basename = "semantic_evaluation_result"
        json.dump(result, open(os.path.join(
            self.model_checkpoint, f"{self.dataset}-{basename}.json"), 'w'))
        # if self.model_weight:
        #     json.dump(result, open(os.path.join(
        #         'results', f"{basename}.json"), 'w'))
        # else:
        #     json.dump(result, open(os.path.join(
        #         self.model_checkpoint, f"{self.dataset}-{basename}.json"), 'w'))


def f1_measure(preds, labels):
    tp = 0
    score = {"precision": 0, "recall": 0, "f1": 0, "turn_acc": 0}
    for p in preds:
        if p in labels:
            tp += 1.0
    if preds:
        score["precision"] = tp/len(preds)
    if labels:
        score["recall"] = tp/len(labels)
    if (score["precision"] + score["recall"]) > 0:
        score["f1"] = 2*(score["precision"]*score["recall"]) / \
            (score["precision"]+score["recall"])
    if tp == len(preds) and tp == len(labels):
        score["turn_acc"] = 1
    return score


def main():
    args = arg_parser()
    eval = Evaluator(args.model_checkpoint,
                     args.dataset,
                     args.model_weight,
                     args.only_action)
    print("model checkpoint", args.model_checkpoint)
    print("generated_file", args.generated_file)
    print("input_file", args.input_file)
    with torch.no_grad():
        if args.do_semantic:
            eval.evaluation(args.input_file)
        if args.do_nlg:
            nlg_result = eval.nlg_evaluation(input_file=args.input_file,
                                             generated_file=args.generated_file,
                                             golden=args.do_golden_nlg)
            if args.generated_file:
                generated_file = args.generated_file
            else:
                generated_file = nlg_result
            eval.evaluation(args.input_file,
                            generated_file)


if __name__ == '__main__':
    main()
