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
    parser.add_argument("--dataset", default="multiwoz")
    parser.add_argument("--golden-emotion", action="store_true",
                        help="golden emotion -> action + utt")
    parser.add_argument("--golden-action", action="store_true",
                        help="golden emotion + action -> utt")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--emotion-mid", action="store_true")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


class Evaluator:
    def __init__(self, model_checkpoint, dataset, model_weight=None, **kwargs):
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.model_weight = model_weight
        self.time = f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
        self.use_sentiment = kwargs.get("use_sentiment", False)
        self.add_persona = kwargs.get("add_persona", False)
        self.emotion_mid = kwargs.get("emotion_mid", False)
        self.emotion_weight = kwargs.get("weight", None)
        self.sample = kwargs.get("sample", False)
        print("self.emotion_weight", self.emotion_weight)

        self.usr = UserActionPolicy(
            model_checkpoint,
            dataset=self.dataset,
            use_sentiment=self.use_sentiment,
            add_persona=self.add_persona,
            emotion_mid=self.emotion_mid,
            weight=self.emotion_weight)

        self.usr.load(os.path.join(model_checkpoint, "pytorch_model.bin"))

        self.r = {"input": [],
                  "golden_acts": [],
                  "golden_utts": [],
                  "golden_emotion": [],
                  "gen_acts": [],
                  "gen_utts": [],
                  "gen_emotion": []}

        if self.use_sentiment:
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

    def generate_results(self, f_eval, golden_emotion=False, golden_action=False):
        emotion_mode = "normal"
        in_file = json.load(open(f_eval))
        mode = "max"
        if self.sample:
            mode = "sample"
        for dialog in tqdm(in_file['dialog']):
            inputs = dialog["in"]
            labels = self.usr._parse_output(dialog["out"])

            if golden_action:
                usr_act = labels["action"]
                usr_emo = labels["emotion"]
                usr_utt = self.usr.generate_text_from_give_semantic(
                    inputs, labels["action"], labels["emotion"])
            elif golden_emotion:
                usr_emo = labels["emotion"]
                output = self.usr.generate_from_emotion(
                    inputs,  emotion=usr_emo, mode=mode)
                output = self.usr._parse_output(output[usr_emo])
                usr_act = self.usr._remove_illegal_action(output["action"])
                usr_utt = output["text"]
            else:
                output = self.usr._parse_output(
                    self.usr._generate_action(inputs, mode=mode, emotion_mode=emotion_mode))
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
                if x not in self.r:
                    self.r[x] = []
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

    def nlg_evaluation(self, input_file=None, generated_file=None, golden_emotion=False, golden_action=False):
        if input_file:
            print("Force generation")
            self.generate_results(input_file, golden_emotion, golden_action)

        elif generated_file:
            self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")
        mode = "max"
        if self.sample:
            mode = "sample"

        nlg_eval = {}
        if golden_action:
            nlg_eval["golden"] = "golden_action"
        elif golden_emotion:
            nlg_eval["golden"] = "golden_emotion"
        else:
            nlg_eval["golden"] = False

        nlg_eval["mode"] = mode
        nlg_eval["emotion_weight"] = self.emotion_weight
        nlg_eval["metrics"] = {}
        nlg_eval["dialog"] = self._transform_result()

        # if golden_action:
        print("Calculate BLEU")
        bleu_metric = load_metric("sacrebleu")
        labels = [[utt] for utt in self.r["golden_utts"]]

        bleu_score = bleu_metric.compute(predictions=self.r["gen_utts"],
                                         references=labels,
                                         force=True)
        print("bleu_metric", bleu_score)
        nlg_eval["metrics"]["bleu"] = bleu_score

        # else:
        print("Calculate SER")
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
            self.r["gen_acts"], self.r["gen_utts"])

        print("{} Missing acts: {}, Total acts: {}, Hallucinations {}, SER {}".format(
            "EmoUSNLG", missing, total, hallucinate, missing/total))
        print(nlg_eval["metrics"])
        nlg_eval["metrics"]["SER"] = missing/total

        # TODO emotion metric

        dir_name = self.model_checkpoint
        json.dump(nlg_eval,
                  open(os.path.join(
                      dir_name, f"{self.time}-nlg_eval.json"), 'w'),
                  indent=2)
        return os.path.join(dir_name, f"{self.time}-nlg_eval.json")

    @staticmethod
    def _intent_domain(action):
        acts = []
        for intent, domain, slot, value in action:
            if [intent, domain] not in acts:
                acts.append([intent, domain])
        return acts

    def evaluation(self, generated_file, golden_emotion=False, golden_action=False):
        # TODO add emotion
        gen_file = json.load(open(generated_file))
        self.read_generated_result(generated_file)

        if golden_action:
            print("golden_action, skip semantic evaluation")
            return

        elif golden_emotion:
            print("golden_emotion, skip emotion evaluation")
            gen_acts, golden_acts = [], []
            for dialog in gen_file['dialog']:
                gen_acts.append(dialog["gen_acts"])
                golden_acts.append(dialog["golden_acts"])
            dialog_result = gen_file['dialog']

        else:
            gen_acts, golden_acts = [], []
            gen_emotions, golden_emotions = [], []
            for dialog in gen_file['dialog']:
                gen_acts.append(dialog["gen_acts"])
                golden_acts.append(dialog["golden_acts"])
                gen_emotions.append(dialog["gen_emotion"])
                golden_emotions.append(dialog["golden_emotion"])
            dialog_result = gen_file['dialog']

        scores = {"complete": {"precision": [], "recall": [], "f1": [], "turn_acc": []},
                  "intent_domain": {"precision": [], "recall": [], "f1": [], "turn_acc": []}}

        # full action
        for gen_act, golden_act in zip(gen_acts, golden_acts):
            s = f1_measure(preds=gen_act, labels=golden_act)
            for metric in scores:
                scores["complete"][metric].append(s[metric])
            s = f1_measure(preds=self._intent_domain(gen_act),
                           labels=self._intent_domain(golden_act))
            for metric in scores:
                scores["intent_domain"][metric].append(s[metric])

        result = {}
        result["emotion_weight"] = self.emotion_weight
        for metric_type, score in scores.items():
            result[metric_type] = {}
            for m, s in score.items():
                result[metric_type][m] = sum(s[m])/len(s[m])
                print(f"{metric_type}-{m}: {result[metric_type][m]}")

        if not golden_emotion:
            emo_score = emotion_score(
                golden_emotions,
                gen_emotions,
                self.model_checkpoint,
                time=self.time,
                no_neutral=False)
            result["emotion"] = {"macro_f1": emo_score["macro_f1"],
                                 "sep_f1": emo_score["sep_f1"]}
            if self.use_sentiment:
                sent_score = sentiment_score(
                    self.r["golden_sentiment"],
                    self.r["gen_sentiment"],
                    self.model_checkpoint,
                    time=self.time)
            else:
                # transfer emotions to sentiment if the model do not generate sentiment
                golden_sentiment = [self.emo2sent[emo]
                                    for emo in golden_emotions]
                gen_sentiment = [self.emo2sent[emo] for emo in gen_emotions]
                sent_score = sentiment_score(
                    golden_sentiment,
                    gen_sentiment,
                    self.model_checkpoint,
                    time=self.time)
            result["sentiment"] = {"macro_f1": sent_score["macro_f1"],
                                   "sep_f1": sent_score["sep_f1"]}

        # for metric in emo_score:
        #     result[metric] = emo_score[metric]
        #     print(f"{metric}: {result[metric]}")

        result["dialog"] = dialog_result

        basename = "semantic_evaluation_result"
        json.dump(
            result,
            open(os.path.join(self.model_checkpoint,
                              f"{self.time}-{self.dataset}-{basename}.json"), 'w'),
            indent=2)


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
                     use_sentiment=args.use_sentiment,
                     emotion_mid=args.emotion_mid,
                     weight=args.weight,
                     sample=args.sample)
    print("model checkpoint", args.model_checkpoint)
    print("generated_file", args.generated_file)
    print("input_file", args.input_file)
    with torch.no_grad():
        if args.generated_file:
            generated_file = args.generated_file
        else:
            nlg_result = eval.nlg_evaluation(input_file=args.input_file,
                                             generated_file=args.generated_file,
                                             golden_emotion=args.golden_emotion,
                                             golden_action=args.golden_action)

            generated_file = nlg_result
        eval.evaluation(generated_file,
                        golden_emotion=args.golden_emotion,
                        golden_action=args.golden_action)


if __name__ == '__main__':
    main()
