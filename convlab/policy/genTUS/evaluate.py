import json
import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import torch
from convlab.nlg.evaluate import fine_SER, get_bleu4
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
                        default="convlab2/policy/genTUS/data/data_validation_v1.py")
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
        dialog_acts, golden_utts, gen_utts = [], [], []
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

            dialog_acts.append(usr_act)
            golden_utts.append(labels["text"])
            gen_utts.append(usr_utt)
        return dialog_acts, golden_utts, gen_utts

    def self_ser(self, f_eval):
        in_file = json.load(open(f_eval))
        dialog_acts, golden_utts, gen_utts = [], [], []
        for dialog in tqdm(in_file['dialog']):
            dialog_acts.append(dialog["predict_action"])
            golden_utts.append(dialog["answer_text"])
            gen_utts.append(dialog["predict_text"])
        return dialog_acts, golden_utts, gen_utts

    def nlg_evaluation(self, input_file=None, generated_file=None, golden=False):
        if input_file:
            print("Force generation")
            dialog_acts, golden_utts, gen_utts = self.generate_results(
                input_file, golden)
            r = {'dialog': []}
            for act, ans, pre in zip(dialog_acts, golden_utts, gen_utts):
                r['dialog'].append({"predict_action": act,
                                    "answer_text": ans,
                                    "predict_text": pre})
                if generated_file:
                    print(f"update result in {generated_file}")
                else:
                    generated_file = os.path.join(
                        self.model_checkpoint, 'generation_results.json')
                    print(f"dump result to {generated_file}")
                json.dump(r, open(generated_file, 'w'), indent=2)

        elif generated_file:
            dialog_acts, golden_utts, gen_utts = self.self_ser(generated_file)
        else:
            print("You must specify the input_file or the generated_file")

        nlg_eval = {
            "dialog_acts": dialog_acts,
            "golden_utts": golden_utts,
            "gen_utts": gen_utts
        }

        print("Calculate SER for golden responses")
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
            nlg_eval["dialog_acts"], nlg_eval["golden_utts"])
        print("Golden response Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
            missing, total, hallucinate, missing/total))

        print("Calculate SER")
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
            nlg_eval["dialog_acts"], nlg_eval["gen_utts"])

        print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
            "genTUSNLG", missing, total, hallucinate, missing/total))
        bleu4 = get_bleu4(nlg_eval["dialog_acts"],
                          nlg_eval["golden_utts"], nlg_eval["gen_utts"])
        print("BLEU-4: %.4f" % bleu4)
        dir_name = self.model_checkpoint
        json.dump(nlg_eval,
                  open(os.path.join(dir_name, "nlg_eval.json"), 'w'))

    def evaluation(self, f_eval):
        in_file = json.load(open(f_eval))
        dialog_result = []
        result = {}
        scores = {"precision": [], "recall": [], "f1": [], "turn_acc": []}
        for dialog in tqdm(in_file['dialog']):
            inputs = dialog["in"]
            labels = self.usr._parse_output(dialog["out"])
            ans_action = self.usr._remove_illegal_action(labels["action"])
            preds = self.usr._generate_action(inputs)
            preds = self.usr._parse_output(preds)
            # print("inputs", inputs)
            # print("goal_list", self.usr.kg.user_goal)
            usr_action = self.usr._remove_illegal_action(preds["action"])
            # print("usr", usr_action)
            # print("ans", ans_action)
            s = f1_measure(preds=usr_action, labels=ans_action)
            for metric in scores:
                scores[metric].append(s[metric])

            print("ans", ans_action)
            print("pre", usr_action)
            d = {"in": inputs,
                 "answer_action": ans_action,
                 "predict_action": usr_action}
            if "text" in preds:
                d["answer_text"] = labels["text"]
                d["predict_text"] = preds["text"]
                # print("pred text", preds["text"])

            dialog_result.append(d)

        for metric in scores:
            result[metric] = sum(scores[metric])/len(scores[metric])
            print(f"{metric}: {result[metric]}")

        result["dialog"] = dialog_result
        basename = "evaluation_result"
        if self.model_weight:
            json.dump(result, open(os.path.join(
                'results', f"{basename}.json"), 'w'))
        else:
            json.dump(result, open(os.path.join(
                self.model_checkpoint, f"{self.dataset}-{basename}.json"), 'w'))


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
    with torch.no_grad():
        if args.do_semantic:
            eval.evaluation(args.input_file)
        if args.do_nlg:
            eval.nlg_evaluation(input_file=args.input_file,
                                generated_file=args.generated_file,
                                golden=args.do_golden_nlg)
            eval.evaluation(args.input_file)


if __name__ == '__main__':
    main()
