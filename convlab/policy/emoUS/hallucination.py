import json
import os
import sys

import torch

from convlab.policy.emoUS.evaluate import Evaluator, arg_parser
from convlab.policy.genTUS.golden_nlg_evaluation import bertnlu_evaluation

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class Hallucinate(Evaluator):
    def __init__(self, model_checkpoint, dataset, model_weight=None, **kwargs):
        super().__init__(model_checkpoint, dataset, model_weight, **kwargs)

    def evaluation(self, input_file="", generated_file="", golden_emotion=False, golden_action=False):
        if input_file:
            print("Force generation")
            self.generate_results(input_file, golden_emotion, golden_action)
        elif generated_file:
            self.read_generated_result(generated_file)
        else:
            print("You must specify the input_file or the generated_file")

        ser, r = bertnlu_evaluation(
            self.r["golden_utt"], self.r["gen_utt"], self.r["golden_act"])
        print(ser)
        with open(os.path.join(self.result_dir, "hallucination"), "w") as f:
            json.dump(r, f, indent=2)


def main():
    args = arg_parser()
    eval = Hallucinate(args.model_checkpoint,
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
