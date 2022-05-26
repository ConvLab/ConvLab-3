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
                positive_keywords = [k.strip() for k in item['keywords+context'].split('\n\n')[0][len("keywords: "):].split('|')[1].split(' : ') if len(k) > 0]
                data["keywords"]["positive_keywords"].append(positive_keywords)
            elif item["keywords+context"].startswith("possible keywords"):
                data["possible keywords"]["predictions"].append(item['predictions'].strip())
                data["possible keywords"]["references"].append(item['response'].strip())
                possible_keywords = [k.strip() for ks in item['keywords+context'].split('\n\n')[0][len("possible keywords: "):].split('|') for k in ks.split(' : ') if len(k) > 0]
                has_positive = True
                for keyword in positive_keywords:
                    if keyword in possible_keywords:
                        possible_keywords.remove(keyword)
                    else:
                        has_positive = False
                        break
                if has_positive:
                    data["possible keywords"]["positive_keywords"].append(positive_keywords)
                else:
                    data["possible keywords"]["positive_keywords"].append([])
                data["possible keywords"]["negative_keywords"].append(possible_keywords)
            # print(data)
            # if len(data["possible keywords"]["positive_keywords"])>0:
            #     break
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
