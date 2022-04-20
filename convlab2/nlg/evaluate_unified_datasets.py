import json
from pprint import pprint
import sacrebleu


def evaluate(predict_result):
    predict_result = json.load(open(predict_result))

    metrics = {}
    predictions, references = [], []
    for sample in predict_result:
        references.append(sample['utterance'])
        predictions.append(sample['predictions']['utterance'])

    metrics['bleu'] = sacrebleu.corpus_bleu(predictions, [references], lowercase=True).score

    return metrics


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate NLU metrics for unified datasets")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args.predict_result)
    pprint(metrics)
