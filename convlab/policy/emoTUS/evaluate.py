import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from datasets import load_metric
# from convlab.policy.genTUS.pg.stepGenTUSagent import \
#     stepGenTUSPG as UserPolicy
from sklearn import metrics
from tqdm import tqdm

from convlab.nlg.evaluate import fine_SER
from convlab.policy.emoTUS.emoTUS import UserActionPolicy

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
    parser.add_argument("--no-neutral", action="store_true",
                        help="skip neutral emotion")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--weight", type=float, default=None)
    return parser.parse_args()


class Evaluator:
    def __init__(self, model_checkpoint, dataset, model_weight=None, only_action=False, use_sentiment=False, weight=None):
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.model_weight = model_weight
        self.time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
        self.use_sentiment = use_sentiment

        self.usr = UserActionPolicy(
            model_checkpoint,
            only_action=only_action,
            dataset=self.dataset,
            use_sentiment=use_sentiment,
            weight=weight)
        self.usr.load(os.path.join(model_checkpoint, "pytorch_model.bin"))

        self.r = {"input": [],
                  "golden_acts": [],
                  "golden_utts": [],
                  "golden_emotion": [],
                  "gen_acts": [],
                  "gen_utts": [],
                  "gen_emotion": []}
        if use_sentiment:
            self.r["golden_sentiment"] = []
            self.r["gen_sentiment"] = []

        sent2emo = json.load(
            open("convlab/policy/emoTUS/sent2emo.json"))
        self.emo2sent = {}
        for sent, emotions in sent2emo.items():
            for emo in emotions:
                self.emo2sent[emo] = sent

    def _append_result(self, temp):
        for x in self.r:
            self.r[x].append(temp[x])

    def generate_results(self, f_eval, golden=False, no_neutral=False):
        emotion_mode = "normal"
        if no_neutral:
            emotion_mode = "no_neutral"
        in_file = json.load(open(f_eval))

        for dialog in tqdm(in_file['dialog']):
            inputs = dialog["in"]
            labels = self.usr._parse_output(dialog["out"])
            if no_neutral and labels["emotion"].lower() == "neutral":
                continue

            if golden:
                usr_act = labels["action"]
                usr_utt = self.usr.generate_text_from_give_semantic(
                    inputs, labels["action"], labels["emotion"])

            else:
                output = self.usr._parse_output(
                    self.usr._generate_action(inputs, emotion_mode=emotion_mode))
                usr_emo = output["emotion"]
                usr_act = self.usr._remove_illegal_action(output["action"])
                usr_utt = output["text"]

            temp = {}
            temp["input"] = inputs
            temp["golden_acts"] = labels["action"]
            temp["golden_utts"] = labels["text"]
            temp["golden_emotion"] = labels["emotion"]

            temp["gen_acts"] = usr_act
            temp["gen_utts"] = usr_utt
            temp["gen_emotion"] = usr_emo

            if self.use_sentiment:
                temp["golden_sentiment"] = labels["sentiment"]
                temp["gen_sentiment"] = output["sentiment"]

            self._append_result(temp)

    def read_generated_result(self, f_eval):
        in_file = json.load(open(f_eval))

        for dialog in tqdm(in_file['dialog']):
            for x in dialog:
                self.r[x].append(dialog[x])

    def _transform_result(self):
        index = [x for x in self.r]
        result = []
        for i in range(len(self.r[index[0]])):
            temp = {}
            for x in index:
                temp[x] = self.r[x][i]
            result.append(temp)
        return result

    def nlg_evaluation(self, input_file=None, generated_file=None, golden=False, no_neutral=False):
        if input_file:
            print("Force generation")
            self.generate_results(input_file, golden, no_neutral)

        elif generated_file:
            self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")

        nlg_eval = {
            "golden": golden,
            "metrics": {},
            "dialog": self._transform_result()
        }

        if golden:
            print("Calculate BLEU")
            bleu_metric = load_metric("sacrebleu")
            labels = [[utt] for utt in self.r["golden_utts"]]

            bleu_score = bleu_metric.compute(predictions=self.r["gen_utts"],
                                             references=labels,
                                             force=True)
            print("bleu_metric", bleu_score)
            nlg_eval["metrics"]["bleu"] = bleu_score

        else:
            print("Calculate SER")
            missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
                self.r["gen_acts"], self.r["gen_utts"])

            print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
                "genTUSNLG", missing, total, hallucinate, missing/total))
            nlg_eval["metrics"]["SER"] = missing/total

            # TODO emotion metric

        dir_name = self.model_checkpoint
        json.dump(nlg_eval,
                  open(os.path.join(
                      dir_name, f"{self.time}-nlg_eval.json"), 'w'),
                  indent=2)
        return os.path.join(dir_name, f"{self.time}-nlg_eval.json")

    def evaluation(self, input_file=None, generated_file=None):
        # TODO add emotion
        force_prediction = True
        if generated_file:
            print("---> use generated file")
            gen_file = json.load(open(generated_file))
            force_prediction = False
            if gen_file["golden"]:
                force_prediction = True
            self.read_generated_result(generated_file)

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
            gen_emotions, golden_emotions = [], []
            for dialog in gen_file['dialog']:
                gen_acts.append(dialog["gen_acts"])
                golden_acts.append(dialog["golden_acts"])
                gen_emotions.append(dialog["gen_emotion"])
                golden_emotions.append(dialog["golden_emotion"])
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
        # TODO no neutral
        emo_score = emotion_score(
            golden_emotions,
            gen_emotions,
            self.model_checkpoint,
            time=self.time,
            no_neutral=False)
        if self.use_sentiment:
            sent_score = sentiment_score(
                self.r["golden_sentiment"],
                self.r["gen_sentiment"],
                self.model_checkpoint,
                time=self.time)
        else:
            # transfer emotions to sentiment if the model do not generate sentiment
            golden_sentiment = [self.emo2sent[emo] for emo in golden_emotions]
            gen_sentiment = [self.emo2sent[emo] for emo in gen_emotions]
            sent_score = sentiment_score(
                golden_sentiment,
                gen_sentiment,
                self.model_checkpoint,
                time=self.time)

        # for metric in emo_score:
        #     result[metric] = emo_score[metric]
        #     print(f"{metric}: {result[metric]}")

        result["dialog"] = dialog_result

        basename = "semantic_evaluation_result"
        json.dump(result, open(os.path.join(
            self.model_checkpoint, f"{self.time}-{self.dataset}-{basename}.json"), 'w'))


def emotion_score(golden_emotions, gen_emotions, dirname=".", time="", no_neutral=False):
    labels = ["Neutral", "Fearful", "Dissatisfied",
              "Apologetic", "Abusive", "Excited", "Satisfied"]
    if no_neutral:
        labels = labels[1:]
    print(labels)
    macro_f1 = metrics.f1_score(golden_emotions, gen_emotions, average="macro")
    sep_f1 = metrics.f1_score(
        golden_emotions, gen_emotions, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_emotions, gen_emotions, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"{time}-emotion.png"))
    r = {"macro_f1": float(macro_f1), "sep_f1": list(
        sep_f1), "cm": [list(c) for c in list(cm)]}
    print(r)
    return r


def sentiment_score(golden_sentiment, gen_sentiment, dirname=".", time=""):
    labels = ["Neutral", "Negative", "Positive"]

    print(labels)
    macro_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average="macro")
    sep_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_sentiment, gen_sentiment, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"{time}-sentiment.png"))
    r = {"macro_f1": float(macro_f1), "sep_f1": list(
        sep_f1), "cm": [list(c) for c in list(cm)]}
    print(r)
    return r


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
                     args.only_action,
                     args.use_sentiment,
                     weight=args.weight)
    print("model checkpoint", args.model_checkpoint)
    print("generated_file", args.generated_file)
    print("input_file", args.input_file)
    with torch.no_grad():
        if args.do_semantic:
            eval.evaluation(args.input_file)
        if args.do_nlg:
            if args.generated_file:
                generated_file = args.generated_file
            else:
                nlg_result = eval.nlg_evaluation(input_file=args.input_file,
                                                 generated_file=args.generated_file,
                                                 golden=args.do_golden_nlg,
                                                 no_neutral=args.no_neutral)

                generated_file = nlg_result
            eval.evaluation(args.input_file,
                            generated_file)


if __name__ == '__main__':
    main()
