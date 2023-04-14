import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from datasets import load_metric
from sklearn import metrics
from tqdm import tqdm
from pprint import pprint

from convlab.nlg.evaluate import fine_SER
from convlab.policy.emoUS.emoUS import UserActionPolicy

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
        self.result_dir = os.path.join(model_checkpoint, "results")
        os.makedirs(self.result_dir, exist_ok=True)
        self.model_weight = model_weight
        self.time = f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
        self.use_sentiment = kwargs.get("use_sentiment", False)
        self.add_persona = kwargs.get("add_persona", False)
        self.emotion_mid = kwargs.get("emotion_mid", False)
        self.emotion_weight = kwargs.get("weight", None)
        self.sample = kwargs.get("sample", False)
        print("self.emotion_weight", self.emotion_weight)
        self.evaluation_result = {
            "emotion prediction": {},
            "semantic action prediction": {},
            "natural language generation": {}}

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
            open("convlab/policy/emoUS/sent2emo.json"))
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

        # save generations
        generations = {}
        generations["time"] = self.time
        generations["golden"] = False
        if golden_action:
            # basically, golden_action includes golden_emotion
            generations["golden"] = "golden_action"
        elif golden_emotion:
            generations["golden"] = "golden_emotion"
        generations["mode"] = mode
        generations["dialog"] = self._transform_result()

        file_name = "generations.json"
        if generations["golden"]:
            file_name = generations['golden'] + "_" + file_name

        with open(os.path.join(self.result_dir, file_name), "w") as f:
            json.dump(generations, f, indent=2)

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

    @staticmethod
    def nlg_evaluation(golden_utts, gen_utts, gen_acts):
        bleu_metric = load_metric("sacrebleu")
        labels = [[utt] for utt in golden_utts]
        bleu_score = bleu_metric.compute(predictions=gen_utts,
                                         references=labels,
                                         force=True)
        missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
            gen_acts, gen_utts)

        return {"bleu": bleu_score["score"], "SER": missing/total}

    @staticmethod
    def _intent_domain(action):
        acts = []
        for intent, domain, slot, value in action:
            if [intent, domain] not in acts:
                acts.append([intent, domain])
        return acts

    def semantic_evaluation(self, gen_acts, golden_acts):
        scores = {"full action": {"precision": [], "recall": [], "f1": [], "turn_acc": []},
                  "intent-domain": {"precision": [], "recall": [], "f1": [], "turn_acc": []}}
        for gen_act, golden_act in zip(gen_acts, golden_acts):
            s = f1_measure(preds=gen_act, labels=golden_act)
            for metric in scores["full action"]:
                scores["full action"][metric].append(s[metric])
            s = f1_measure(preds=self._intent_domain(gen_act),
                           labels=self._intent_domain(golden_act))
            for metric in scores["intent-domain"]:
                scores["intent-domain"][metric].append(s[metric])

        result = {}
        for metric_type, score in scores.items():
            result[metric_type] = {}
            for m, s in score.items():
                result[metric_type][m] = sum(s)/len(s)
        return result

    def evaluation(self, input_file="", generated_file="", golden_emotion=False, golden_action=False):
        if input_file:
            print("Force generation")
            self.generate_results(input_file, golden_emotion, golden_action)
        elif generated_file:
            self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")

        r = self.nlg_evaluation(
            self.r["golden_utts"], self.r["gen_utts"], self.r["gen_acts"])
        for metric, score in r.items():
            self.evaluation_result["natural language generation"][metric] = score

        if not golden_action:
            r = self.semantic_evaluation(
                self.r["gen_acts"], self.r["golden_acts"])
            for metric, score in r.items():
                self.evaluation_result["semantic action prediction"][metric] = score

        if not golden_emotion and not golden_action:
            r = emotion_score(self.r["golden_emotion"],
                              self.r["gen_emotion"],
                              self.result_dir)
            self.evaluation_result["emotion prediction"]["emotion"] = {}
            self.evaluation_result["emotion prediction"]["emotion"]["macro_f1"] = r["macro_f1"]
            self.evaluation_result["emotion prediction"]["emotion"]["sep_f1"] = {
                emo: f1 for emo, f1 in zip(r["label"], r["sep_f1"])}

            if self.use_sentiment:
                golden_sentiment = self.r["golden_sentiment"]
                gen_sentiment = self.r["gen_sentiment"]
            else:
                # transfer emotions to sentiment if the model do not generate sentiment
                golden_sentiment = [self.emo2sent[emo]
                                    for emo in self.r["golden_emotion"]]
                gen_sentiment = [self.emo2sent[emo]
                                 for emo in self.r["gen_emotion"]]
            r = sentiment_score(golden_sentiment,
                                gen_sentiment,
                                self.result_dir)

            self.evaluation_result["emotion prediction"]["sentiment"] = {}
            self.evaluation_result["emotion prediction"]["sentiment"]["macro_f1"] = r["macro_f1"]
            self.evaluation_result["emotion prediction"]["sentiment"]["sep_f1"] = {
                emo: f1 for emo, f1 in zip(r["label"], r["sep_f1"])}

        pprint(self.evaluation_result)

    # def save_results(self):

    # def print_result(self):
    #     print("=== Natural language generation ===")
    #     print("Sacre-BLEU", nlg_eval["metrics"]["bleu"]["score"])
    #     print("SER", nlg_eval["metrics"]["SER"])
    #     self.r[""]


def emotion_score(golden_emotions, gen_emotions, dirname=".", no_neutral=False):
    labels = ["Neutral", "Fearful", "Dissatisfied",
              "Apologetic", "Abusive", "Excited", "Satisfied"]
    if no_neutral:
        labels = labels[1:]

    macro_f1 = metrics.f1_score(golden_emotions, gen_emotions, average="macro")
    sep_f1 = metrics.f1_score(
        golden_emotions, gen_emotions, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_emotions, gen_emotions, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"emotion.png"))
    r = {"label": labels,
         "macro_f1": float(macro_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
    return r


def sentiment_score(golden_sentiment, gen_sentiment, dirname="."):
    labels = ["Neutral", "Negative", "Positive"]

    macro_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average="macro")
    sep_f1 = metrics.f1_score(
        golden_sentiment, gen_sentiment, average=None, labels=labels)
    cm = metrics.confusion_matrix(
        golden_sentiment, gen_sentiment, normalize="true", labels=labels)
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(dirname, f"sentiment.png"))
    r = {"label": labels,
         "macro_f1": float(macro_f1),
         "sep_f1": list(sep_f1),
         "cm": [list(c) for c in list(cm)]}
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
    print("=== evaluation ===")
    print("model checkpoint", args.model_checkpoint)
    print("generated_file", args.generated_file)
    print("input_file", args.input_file)
    with torch.no_grad():
        eval.evaluation(input_file=args.input_file,
                        generated_file=args.generated_file,
                        golden_emotion=args.golden_emotion,
                        golden_action=args.golden_action)


if __name__ == '__main__':
    main()
