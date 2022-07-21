from tabulate import tabulate
import os
import json
from tqdm import tqdm
from datasets import load_metric
import numpy as np

def evaluate(filename, metric):
    """
    It reads the predictions, references, and knowledge from a file, and then computes the metric
    
    :param filename: the path to the file containing the predictions
    :param metric: the metric to use for evaluation
    :return: The result of the evaluation.
    """
    predictions, references, knowledge = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            predictions.append(item['predictions'])
            references.append(item['response'])
            knowledge.append(item['knowledge'])
    result = metric.compute(predictions=predictions, references=references, knowledge=knowledge)
    return result


def avg_result(results):
    """
    It takes a list of dictionaries, and returns a dictionary with the same keys, but the values are the
    mean and standard deviation of the values in the input dictionaries
    
    :param results: a list of dictionaries, each dictionary is the result of a single run of the model
    :return: The average and standard deviation of the results.
    """
    ret = {}
    for k in results[0]:
        m = round(np.mean([result[k] for result in results]), 2)
        v = round(np.std([result[k] for result in results], ddof=1), 2) if len(results) > 1 else None
        ret[k] = f"{m}({v})"
    return ret


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="create data for seq2seq training")
    parser.add_argument("--output_dirs", type=str, nargs='*', required=True)
    parser.add_argument('--tasks', '-t', type=str, nargs='*', choices=['nlg', 'kvret', 'opendialkg', 'personachat', 'wow'], help='names of tasks')
    parser.add_argument('--shots', '-s', type=int, nargs='*', help='how many data is used for training and evaluation, ratio if < 1 else absolute number')
    parser.add_argument('--dial_ids_orders', '-o', type=int, nargs='*', help='which data order is used for experiments')
    args = parser.parse_args()
    print(args)
    
    tables = []
    for task_name in tqdm(args.tasks, desc='tasks'):
        metric = load_metric("metric.py", task_name)
        dataset_name = task_name if task_name != "nlg" else "multiwoz21"
        for shot in tqdm(args.shots, desc='shots'):
            for output_dir in tqdm(args.output_dirs, desc='models'):
                model_name = output_dir.split('/')[-1]
                if task_name == "wow":
                    test_splits = ["_seen", "_unseen"]
                else:
                    test_splits = [""]
                for test_split in test_splits:
                    results = []
                    for dial_ids_order in tqdm(args.dial_ids_orders, desc='dial_ids_orders'):
                        filename = os.path.join(output_dir, task_name, f"{dataset_name}_{shot}shot_order{dial_ids_order}/gen{test_split}/generated_predictions.json")
                        results.append(evaluate(filename, metric))
                    res = {
                        "dataset": f"{task_name}-{shot}shot",
                        "model": f"{model_name}{test_split}",
                        **avg_result(results)
                    }
                    tables.append(res)
                    # print(res)
    res = tabulate(tables, headers='keys', tablefmt='github')
    with open(f'eval_results.txt', 'w', encoding='utf-8') as f:
        print(res, file=f)
