import json
import os
import pandas as pd
from convlab.policy.emoUS.evaluate import Evaluator


def main():
    structures = json.load(open("convlab/policy/emoUS/structure.json"))
    result = {"model": [], "weight": []}
    for s in structures:
        for weight in [0.98, 0.95, 0.9, 0.85, 0.8]:
            generated_file = os.path.join(
                s["model"], f"weight-{weight}", f"{s['prefix']}-generations.json")
            eval = Evaluator(s["model"],
                             "multiwoz",
                             "",
                             use_sentiment=s["use_sentiment"],
                             emotion_mid=s["emotion_mid"],
                             weight=weight,
                             sample=False)
            r = eval.evaluation(generated_file=generated_file)
            result["model"].append(s["label"])
            result["weight"].append(weight)
            for k, v in r.items():
                if k == "emotion prediction":
                    for x in ["sentiment", "emotion"]:
                        if f"{x}_f1" not in result:
                            result[f"{x}_f1"] = []
                        result[f"{x}_f1"].append(v[x]["macro_f1"])
                        for emo, f1 in v[x]["sep_f1"]:
                            if f"{x}_{emo}" not in result:
                                result[f"{x}_{emo}"] = []
                            result[f"{x}_{emo}"].append(f1)
    pd.DataFrame.from_dict(result).to_csv("result.csv")


if __name__ == "__main__":
    main()
