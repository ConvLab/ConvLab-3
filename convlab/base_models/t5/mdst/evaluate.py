import os
import json
from pprint import pprint
from tabulate import tabulate
from convlab.util import load_ontology
from convlab.base_models.t5.mdst.utils import evaluate_and_write_dial, label_dial


def evaluate(args):
    full_state = json.load(open(os.path.join('data/'+'/'.join(args.predict_result.split('/')[1:-3]), 'full_state.json')))
    predict_result = json.load(open(args.predict_result))
    if args.origin_data:
        ori_data = json.load(open(args.origin_data))
    else:
        ori_data = predict_result

    metrics_by_domains = {}
    cross_domain_dials_metrics = {'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []}
    dials_metrics_by_dom_cnt = {i: {'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []} for i in range(2)}

    for dial, dial_ori in zip(predict_result, ori_data):
        label_dial(dial_ori, ignore_active_domain=True)
        for turn_idx in range(0,len(dial_ori['turns']),2):
            for k in ['cross_domain', 'cross_domain_slot_value', 'delta_c']:
                dial['turns'][turn_idx][k] = dial_ori['turns'][turn_idx][k]

        dial_domains = tuple(sorted(dial['domains']))
        if len(dial_domains) not in dials_metrics_by_dom_cnt:
            dials_metrics_by_dom_cnt[len(dial_domains)] = {'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []}
        metrics_by_domains.setdefault(dial_domains, {'total dials': 0, 'cross_domain dials': 0, 
            'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []})
        metrics_by_domains[dial_domains]['total dials'] += 1
        cross_domain = False

        evaluate_and_write_dial(dial, full_state)
        JGA, RSA, TA, CDTA = [], [], [], []
        domain_cnt = []
        actived_domains = []
        for turn_idx in range(0,len(dial['turns']),2):
            turn = dial['turns'][turn_idx]
            if turn['cross_domain']:
                cross_domain = True
            for domain in turn['active_domains']:
                if domain not in actived_domains:
                    actived_domains.append(domain)
            domain_cnt.append(len(actived_domains))
            # for domain, metric in turn['active_domains_metrics'].items():
            #     domain_cnt = actived_domains.index(domain)
            #     dials_metrics_by_dom_cnt[domain_cnt]['JGA'].append(metric['JGA'])
            #     if metric['RSA'] is not None:
            #         dials_metrics_by_dom_cnt[domain_cnt]['RSA'].append(metric['RSA'])
            #     dials_metrics_by_dom_cnt[domain_cnt]['TA'].append(metric['TA'])
            #     if turn['cross_domain']:
            #         dials_metrics_by_dom_cnt[domain_cnt]['CDTA'].append(metric['TA'])
            JGA.append(turn['metrics']['JGA'])
            RSA.append(turn['metrics']['RSA'])
            TA.append(turn['metrics']['TA'])
            if turn['cross_domain']:
                CDTA.append(turn['metrics']['TA'])
            else:
                CDTA.append(None)
        for k, v in zip(['JGA', 'RSA', 'TA', 'CDTA'], [JGA, RSA, TA, CDTA]):
            for c, m in zip(domain_cnt, v):
                if m is not None:
                    dials_metrics_by_dom_cnt[c][k].append(m)
            v = [ele for ele in v if ele is not None]
            metrics_by_domains[dial_domains][k].extend(v)
            if cross_domain:
                cross_domain_dials_metrics[k].extend(v)
        metrics_by_domains[dial_domains]['cross_domain dials'] += cross_domain
    
    if 'sgd' in args.predict_result:
        sgd10_domains = set(['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1'])
        sgd15_domains = set(['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
                            'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2'])
        aggregate = {k: {'total dials': 0, 'cross_domain dials': 0, 'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []} for k in ['sgd10', 'sgd15', 'sgd19', 'all']}
        for domains in metrics_by_domains:
            for metric in metrics_by_domains[domains]:
                aggregate['all'][metric] += metrics_by_domains[domains][metric]
                if all([domain in sgd10_domains for domain in domains]):
                    aggregate['sgd10'][metric] += metrics_by_domains[domains][metric]
                elif all([domain in sgd15_domains for domain in domains]):
                    aggregate['sgd15'][metric] += metrics_by_domains[domains][metric]
                else:
                    aggregate['sgd19'][metric] += metrics_by_domains[domains][metric]
    else:
        aggregate = {'all': {'total dials': 0, 'cross_domain dials': 0, 'JGA': [], 'RSA': [], 'TA': [], 'CDTA': []}}
        for domains in metrics_by_domains:
            for metric in metrics_by_domains[domains]:
                aggregate['all'][metric] += metrics_by_domains[domains][metric]

    table = []
    for domains, metrics in sorted(metrics_by_domains.items(), key=lambda x: x[1]['cross_domain dials']/x[1]['total dials']) \
            + [(f'domain_{i}', dials_metrics_by_dom_cnt[i]) for i in dials_metrics_by_dom_cnt] \
            + list(aggregate.items()) + [('cross-domain', cross_domain_dials_metrics)]:
        metrics['total turns'] = len(metrics['JGA'])
        metrics['cross-domain turns'] = len(metrics['CDTA'])
        metrics['JGA'] = sum(metrics['JGA'])/len(metrics['JGA']) if len(metrics['JGA']) else 'NA'
        metrics['RSA'] = sum(metrics['RSA'])/len(metrics['RSA']) if len(metrics['RSA']) else 'NA'
        metrics['TA'] = sum(metrics['TA'])/len(metrics['TA']) if len(metrics['TA']) else 'NA'
        metrics['CDTA'] = sum(metrics['CDTA'])/len(metrics['CDTA']) if len(metrics['CDTA']) else 'NA'

        table.append({'domains': domains, **metrics})
    res = tabulate(table, headers='keys', tablefmt='github', floatfmt='.4f')
    output_file = args.predict_result.replace('generated_predictions', 'result').replace('.json', '_ori_cdta.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        print(res, file=f)
        
    output_file = args.predict_result.replace('generated_predictions', 'eval_predictions')
    json.dump(predict_result, open(output_file, 'w', encoding='utf-8'), indent=2)

    return aggregate


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="calculate DST metrics for unified datasets")
    parser.add_argument('--dataset_name', '-d', metavar='dataset_name', help='names of unified dataset')
    parser.add_argument('--predict_result', '-p', type=str, required=True, help='path to the prediction file that in the unified data format')
    parser.add_argument('--origin_data', '-i', type=str, help='path to the origin test file that in the unified data format')
    args = parser.parse_args()
    print(args)
    metrics = evaluate(args)
    pprint(metrics)
