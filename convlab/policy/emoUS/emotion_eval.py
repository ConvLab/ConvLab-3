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

from convlab.nlg.evaluate import fine_SER
from convlab.policy.emoUS.emoUS import UserActionPolicy

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, help="the model path")
    parser.add_argument("--input-file", type=str, help="the testing input file",
                        default="")
    parser.add_argument("--generated-file", type=str, help="the generated results",
                        default="")
    parser.add_argument("--dataset", default="multiwoz")

    # model parameter
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--emotion-mid", action="store_true")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


class Evaluator:
    def __init__(self, model_checkpoint, dataset,  **kwargs):
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint

        self.time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
        self.use_sentiment = kwargs.get("use_sentiment", False)
        self.add_persona = kwargs.get("add_persona", True)
        self.emotion_mid = kwargs.get("emotion_mid", False)
        weight = kwargs.get("weight", None)
        self.sample = kwargs.get("sample", False)

        self.usr = UserActionPolicy(
            model_checkpoint,
            dataset=self.dataset,
            use_sentiment=self.use_sentiment,
            add_persona=self.add_persona,
            emotion_mid=self.emotion_mid,
            weight=weight)

        self.usr.load(os.path.join(model_checkpoint, "pytorch_model.bin"))

        """
        self.r = {"input", "golden_acts", "golden_utts", "golden_emotions",
        emotion_acts, emotion_utts}
        """

        self.r = {"input": [],
                  "golden_acts": [],
                  "golden_utts": [],
                  "golden_emotion": []}

        if self.use_sentiment:
            self.r["golden_sentiment"] = []
            self.r["gen_sentiment"] = []

        self.emotion_list = []

        for emotion in json.load(open("convlab/policy/emoUS/emotion.json")):
            self.emotion_list.append(emotion)
            self.r[f"{emotion}_acts"] = []
            self.r[f"{emotion}_utts"] = []

        sent2emo = json.load(
            open("convlab/policy/emoUS/sent2emo.json"))
        self.emo2sent = {}
        for sent, emotions in sent2emo.items():
            for emo in emotions:
                self.emo2sent[emo] = sent

    def _append_result(self, temp):
        for x in self.r:
            self.r[x].append(temp[x])

    def generate_results(self, f_eval, golden=False):
        emotion_mode = "normal"
        in_file = json.load(open(f_eval))

        for dialog in tqdm(in_file['dialog']):
            temp = {}
            inputs = dialog["in"]
            labels = self.usr._parse_output(dialog["out"])

            response = self.usr.generate_from_emotion(
                raw_inputs=inputs)

            temp["input"] = inputs
            temp["golden_acts"] = labels["action"]
            temp["golden_utts"] = labels["text"]
            temp["golden_emotion"] = labels["emotion"]

            for emotion, resp in response.items():
                output = self.usr._parse_output(resp)
                temp[f"{emotion}_acts"] = output["action"]
                temp[f"{emotion}_utts"] = output["text"]

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

    def nlg_evaluation(self, input_file=None, generated_file=None, golden=False):
        if input_file:
            print("Force generation")
            self.generate_results(input_file, golden)

        elif generated_file:
            self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")
        mode = "max"
        if self.sample:
            mode = "sample"

        nlg_eval = {
            "golden": golden,
            "mode": mode,
            "metrics": {},
            "dialog": self._transform_result()
        }

        # TODO emotion metric

        dir_name = self.model_checkpoint
        json.dump(nlg_eval,
                  open(os.path.join(
                      dir_name, f"{self.time}-nlg_eval.json"), 'w'),
                  indent=2)
        return os.path.join(dir_name, f"{self.time}-nlg_eval.json")

    def evaluation(self, input_file=None, generated_file=None):
        # TODO add emotion
        gen_file = json.load(open(generated_file))
        self.read_generated_result(generated_file)

        r = {"golden_acts": [], "golden_emotions": [], "golden_utts": []}
        for emotion in self.emotion_list:
            r[f"{emotion}_acts"] = []
            r[f"{emotion}_utts"] = []

        for dialog in gen_file['dialog']:
            r["golden_acts"].append(dialog["golden_acts"])
            r["golden_emotions"].append(dialog["golden_emotion"])
            r["golden_utts"].append(dialog["golden_utts"])
            for emotion in self.emotion_list:
                r[f"{emotion}_acts"].append(dialog[f"{emotion}_acts"])
                r[f"{emotion}_utts"].append(dialog[f"{emotion}_utts"])

        dialog_result = gen_file['dialog']

        scores = {}
        for emotion in self.emotion_list:
            # if emotion == "Neutral":
            #     continue
            scores[emotion] = {"precision": [],
                               "recall": [], "f1": [], "turn_acc": []}
            for gen_act, golden_act in zip(r[f"{emotion}_acts"], r["Neutral_acts"]):
                s = f1_measure(preds=gen_act, labels=golden_act)
                for metric in scores[emotion]:
                    scores[emotion][metric].append(s[metric])

        result = {}
        for emotion in self.emotion_list:
            # if emotion == "Neutral":
            #     continue
            result[emotion] = {}
            for metric in scores[emotion]:
                result[emotion][metric] = sum(
                    scores[emotion][metric])/len(scores[emotion][metric])
            result[emotion]["bleu"] = bleu(golden_utts=r["Neutral_utts"],
                                           gen_utts=r[f"{emotion}_utts"])
            result[emotion]["SER"] = SER(gen_utts=r[f"{emotion}_utts"],
                                         gen_acts=r[f"{emotion}_acts"])

            result[emotion]["len"] = avg_len(gen_utts=r[f"{emotion}_utts"])

            rouge_score = rouge(golden_utts=r["Neutral_utts"],
                                gen_utts=r[f"{emotion}_utts"])
            for metric, score in rouge_score.items():
                result[emotion][metric] = score.mid.fmeasure

            print("emotion:", emotion)
            for metric in result[emotion]:
                print(f"{metric}: {result[emotion][metric]}")

        # for metric in emo_score:
        #     result[metric] = emo_score[metric]
        #     print(f"{metric}: {result[metric]}")

        result["dialog"] = dialog_result

        basename = "semantic_evaluation_result"
        json.dump(result, open(os.path.join(
            self.model_checkpoint, f"{self.time}-{self.dataset}-{basename}.json"), 'w'), indent=2)


def avg_len(gen_utts):
    n = [len(s.split()) for s in gen_utts]
    return sum(n)/len(n)


def bleu(golden_utts, gen_utts):
    bleu_metric = load_metric("sacrebleu")
    labels = [[utt] for utt in golden_utts]

    bleu_score = bleu_metric.compute(predictions=gen_utts,
                                     references=labels,
                                     force=True)
    return bleu_score["score"]


def rouge(golden_utts, gen_utts):
    rouge_metric = load_metric("rouge")
    rouge_score = rouge_metric.compute(predictions=gen_utts,
                                       references=golden_utts)
    return rouge_score


def SER(gen_utts, gen_acts):
    missing, hallucinate, total, hallucination_dialogs, missing_dialogs = fine_SER(
        gen_acts, gen_utts)
    if total <= 0:
        print("ERROR, total = 0")
        return 1
    return missing/total


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
                     use_sentiment=args.use_sentiment,
                     emotion_mid=args.emotion_mid,
                     weight=args.weight,
                     sample=args.sample)
    print("=== evaluation ===")
    print("model checkpoint", args.model_checkpoint)
    print("generated_file", args.generated_file)
    print("input_file", args.input_file)
    with torch.no_grad():
        if args.generated_file:
            generated_file = args.generated_file
        else:
            nlg_result = eval.nlg_evaluation(input_file=args.input_file,
                                             generated_file=args.generated_file)

            generated_file = nlg_result
        eval.evaluation(args.input_file,
                        generated_file)


if __name__ == '__main__':
    main()
