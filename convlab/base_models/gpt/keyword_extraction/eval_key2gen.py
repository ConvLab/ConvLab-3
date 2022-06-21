import json
import datasets
from tabulate import tabulate

def main(predict_result):
    data = {
        "grounded keywords": {
            "positive_keywords": [], "negative_keywords": None,
            "predictions": [], "references": []
        },
        "all keywords": {
            "positive_keywords": [], "negative_keywords": [],
            "predictions": [], "references": []
        },
        "no keywords": {
            "positive_keywords": None, "negative_keywords": None,
            "predictions": [], "references": []
        }
    }
    with open(predict_result) as f:
        for line in f:
            item = json.loads(line)
            prediction = item['predictions'].strip()
            reference = item['target'].strip()
            if 'all_keywords' in item and item['all_keywords']:
                sample_type = 'all keywords'

                positive_keywords = [k for g in item['keywords'] for k in g]
                data[sample_type]["positive_keywords"].append(positive_keywords)

                all_keywords = [k for g in item['all_keywords'] for k in g]
                for keyword in positive_keywords:
                    all_keywords.remove(keyword)
                data[sample_type]["negative_keywords"].append(all_keywords)

            elif 'keywords' in item and item['keywords']:
                sample_type = 'grounded keywords'

                positive_keywords = [k for g in item['keywords'] for k in g]
                data[sample_type]["positive_keywords"].append(positive_keywords)
            
            else:
                sample_type = 'no keywords'

            data[sample_type]["predictions"].append(prediction)
            data[sample_type]["references"].append(reference)

    metric = datasets.load_metric('./key2gen_metric.py')
    table = []
    for sample_type in data:
        table.append({'sample_type': sample_type, **metric.compute(**data[sample_type])})
    print(tabulate(table, headers='keys', tablefmt='github'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="evaluate keywords to response generation performance")
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the output file generated_predictions.json')
    args = parser.parse_args()
    print(args)
    main(args.predict_result)
