import json
import datasets
from tabulate import tabulate

def main(predict_result):
    data = {
        "keywords": {
            "positive_keywords": [], "negative_keywords": None,
            "predictions": [], "references": []
        },
        "possible keywords": {
            "positive_keywords": [], "negative_keywords": [],
            "predictions": [], "references": []
        }
    }
    with open(predict_result) as f:
        for line in f:
            item = json.loads(line)
            if item["keywords+context"].startswith("keywords"):
                data["keywords"]["predictions"].append(item['predictions'].strip())
                data["keywords"]["references"].append(item['response'].strip())
                positive_keywords = [k for k in item['keywords+context'].split('\n\n')[0][len("keywords: "):].split(' | ') if len(k) > 0]
                data["keywords"]["positive_keywords"].append(positive_keywords)
            elif item["keywords+context"].startswith("possible keywords"):
                data["possible keywords"]["predictions"].append(item['predictions'].strip())
                data["possible keywords"]["references"].append(item['response'].strip())
                possible_keywords = [k for k in item['keywords+context'].split('\n\n')[0][len("possible keywords: "):].split(' | ') if len(k) > 0]
                for keyword in positive_keywords:
                    possible_keywords.remove(keyword)
                data["possible keywords"]["positive_keywords"].append(positive_keywords)
                data["possible keywords"]["negative_keywords"].append(possible_keywords)
    metric = datasets.load_metric('./key2gen_metric.py')
    table = [{'prompt': "keywords", **metric.compute(**data["keywords"])}]
    if len(data["possible keywords"]["predictions"]) > 0:
        table.append({'prompt': "possible keywords", **metric.compute(**data["possible keywords"])})
    print(tabulate(table, headers='keys', tablefmt='github'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="evaluate keywords to response generation performance")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the output file generated_predictions.json')
    args = parser.parse_args()
    print(args)
    main(args.predict_result)
