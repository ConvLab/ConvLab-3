"""split data by single domain or multi-domain"""
import os
import json
import random
import re
from tqdm import tqdm
from copy import deepcopy
from tabulate import tabulate
from collections import Counter
from convlab.util import load_dataset
from convlab.base_models.t5.mdst.utils import label_dial


def get_statistics(dataset_name):
    data = load_dataset(dataset_name)
    table = []
    domain_cnt = {}
    for data_split in data:
        for dial in data[data_split]:
            if 'police' in dial['domains'] or 'hospital' in dial['domains']:
                continue
            domains = sorted(set(dial['domains']) - set(['general']))
            domain_cnt.setdefault(tuple(domains), {'train': 0, 'validation': 0, 'test': 0})
            domain_cnt[tuple(domains)][data_split] += 1
    for domains, stat in sorted(domain_cnt.items(), key=lambda x:len(x[0])*10000+sum(x[1].values())):
        s = sum(stat.values())
        res = {'domains':domains, 'all': s}
        for data_split in data:
            res[data_split] = stat[data_split]
        table.append(res)
    return table


def simplify_dial(dial):
    new_dial = {'domains': dial['domains'], 'turns': []}
    for turn in dial['turns']:
        new_dial['turns'].append({
            'speaker': turn['speaker'],
            'utterance': turn['utterance']
        })
        if turn['speaker'] != 'user':
            continue
        
        state = deepcopy(turn['state'])
        for domain in list(state.keys()):
            if domain not in new_dial['domains']:
                # remove error domain
                state.pop(domain)
                continue
            for slot in list(state[domain].keys()):
                if "18|8 Fine Men'S Salons - Lafayette" in state[domain][slot]:
                    state[domain][slot] = "18|8 Fine Men'S Salons - Lafayette"
                    continue

                vs = re.split('(?<=\S)\|(?=\S)', state[domain][slot])
                if len(vs[0]) == 0:
                    state[domain].pop(slot)
                    continue
                # only the first variation of value
                state[domain][slot] = vs[0]
                assert state[domain][slot] == state[domain][slot].strip()
            if len(state[domain]) == 0:
                state.pop(domain)
        new_dial['turns'][-1]['state'] = state
   
    return new_dial


def split_data(data_by_domain, train_ratio=0.8, val_ratio=0.1):
    train_data = []
    validation_data = []
    test_data = []
    for domains in data_by_domain:
        random.shuffle(data_by_domain[domains])
        total = len(data_by_domain[domains])
        train_data.extend(data_by_domain[domains][:round(total*train_ratio)])
        validation_data.extend(data_by_domain[domains][round(total*train_ratio):round(total*(train_ratio+val_ratio))])
        test_data.extend(data_by_domain[domains][round(total*(train_ratio+val_ratio)):])
    return train_data, validation_data, test_data


def get_cross_domain_slot_pairs(multi_domain_dials, full_state):
    # get coref (same value for different slots) in multi-domain dialog
    # P(tgt_slot==src_slot | src_slot is not empty) > 1%
    # many have false positive (not 100% precise) like "stars" and "people"
    slot_pairs = {(src_d, src_s, tgt_d, tgt_s): [0, 0] 
                  for src_d in full_state for src_s in full_state[src_d] 
                  for tgt_d in full_state if tgt_d != src_d for tgt_s in full_state[tgt_d]}
    for dial in multi_domain_dials:
        domain_pairs = set()
        prev_state = {}
        for turn_idx in range(0, len(dial['turns']), 2):
            state_update = dial['turns'][turn_idx]['state_update']
            if len(state_update) != 0:
                # non-empty state update, may from all previous domains
                for tgt_d in state_update:
                    for src_d in prev_state:
                        if src_d != tgt_d:
                            domain_pairs.add((src_d, tgt_d))
            prev_state = dial['turns'][turn_idx]['state']

        # last turn state
        final_state = dial['turns'][-2]['state']

        for src_d, tgt_d in domain_pairs:
            if src_d not in final_state or tgt_d not in final_state:
                # error annotation
                continue
            for src_s, src_v in final_state[src_d].items():
                for tgt_s in full_state[tgt_d]:
                    if tgt_s not in final_state[tgt_d]:
                        tgt_v = ''
                    else:
                        tgt_v = final_state[tgt_d][tgt_s]
                    assert src_v != ''
                    if src_v == tgt_v:
                        # cnt for value transfer
                        slot_pairs[(src_d, src_s, tgt_d, tgt_s)][0] += 1
                    # cnt for all cases
                    slot_pairs[(src_d, src_s, tgt_d, tgt_s)][1] += 1
                    
    slot_pairs = {str(k):v for k,v in slot_pairs.items() if v[0]>(v[1]*0.1) and v[1]>0}

    return slot_pairs


def create_dst_data(dataset_name, data_dir, args):
    random.seed(42)
    data_by_split = load_dataset(dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    table = get_statistics(dataset_name)
    res = tabulate(table, headers='keys', tablefmt='github')
    with open(f'{data_dir}/original_data_stat.md', 'w', encoding='utf-8') as f:
        print(res, file=f)

    if dataset_name == 'multiwoz21':
        groups = [['attraction', 'restaurant', 'hotel', 'train', 'taxi']]
    elif 'sgd' in dataset_name:
        groups = [
            ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
             'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2', 'Alarm_1', 'Buses_3', 'Movies_1', 'RentalCars_3'],
            ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1', \
             'Calendar_1', 'Homes_2', 'Music_3', 'Restaurants_1', 'RideSharing_2'],
            ['Banks_1', 'Events_3', 'Flights_4', 'Hotels_4', 'Media_3', 'Payment_1', 'Services_1', 'Trains_1', 'Travel_1', 'Weather_1']
            ]
        groups = [[service+dataset_name[3:] for service in group] for group in groups]

    group_stats = []
    for group_idx, domain_group in enumerate(groups):
        data_splits = data_by_split.keys()
        single_domain = {domain:[] for domain in domain_group}
        multi_domain = {}
        # split by single/multi-domain, find coref state
        for data_split in data_splits:
            for dial in data_by_split[data_split]:
                domains = dial['domains']
                if dataset_name == 'multiwoz21':
                    if 'police' in domains or 'hospital' in domains:
                        continue
                    if 'general' in domains:
                        domains.remove('general')
                elif 'sgd' in dataset_name:
                    if any([domain not in domain_group for domain in domains]):
                        continue
                if len(domains) == 1:
                    domain = domains[0]
                    single_domain.setdefault(domain, [])
                    single_domain[domain].append(simplify_dial(dial))
                else:
                    domains = tuple(sorted(domains))
                    multi_domain.setdefault(domains, [])
                    multi_domain[domains].append(simplify_dial(dial))

        stats = {}
        single_samples, multi_samples = 0, 0
        
        single_domain_data = []
        multi_domain_data = []
        # ignore multi-domain combinations that have least than 10 dials
        # label turn delta_c, cross_domain
        delta_c_cnt = {'single': Counter(), 'multi': Counter()}
        cross_domain_cnt = {'single': Counter(), 'multi': Counter()}
        cross_domain_dial_cnt = 0
        last_2turn_no_update = 0
        for domain in single_domain:
            stats[domain] = len(single_domain[domain])
            single_domain_data.extend(single_domain[domain])
            for dial in single_domain[domain]:
                label_dial(dial)
                if len(dial['turns'][-2]['state_update']) == 0:
                    last_2turn_no_update += 1
                for turn_idx in range(0, len(dial['turns']),2):
                    turn = dial['turns'][turn_idx]
                    delta_c, cross_domain = turn['delta_c'], turn['cross_domain']
                    assert cross_domain is False or delta_c > 1
                    delta_c_cnt['single'][delta_c] += 1
                    cross_domain_cnt['single'][cross_domain] += 1

                single_samples += len(dial['turns']) >> 1
        for domains, data in sorted(multi_domain.items(), key=lambda x:len(x[1])):
            if len(data) < 10:
                multi_domain.pop(domains)
            stats[domains] = len(data)
            multi_domain_data.extend(data)
            for dial in data:
                label_dial(dial)
                cross_domain_dial = False
                for turn_idx in range(0, len(dial['turns']),2):
                    turn = dial['turns'][turn_idx]
                    delta_c, cross_domain = turn['delta_c'], turn['cross_domain']
                    assert cross_domain is False or delta_c > 1
                    delta_c_cnt['multi'][delta_c] += 1
                    cross_domain_cnt['multi'][cross_domain] += 1
                    if cross_domain:
                        cross_domain_dial = True
                if cross_domain_dial:
                    cross_domain_dial_cnt += 1

                multi_samples += len(dial['turns']) >> 1

        stats['multi-domain combinations'] = len(multi_domain)
        stats['sinlge-domain dials'] = len(single_domain_data)
        stats['sinlge-domain last turn no update'] = last_2turn_no_update
        stats['multi-domain dials'] = len(multi_domain_data)
        stats['multi-domain cross-domain dials'] = cross_domain_dial_cnt/len(multi_domain_data)
        stats['total dials'] = stats['multi-domain dials'] + stats['sinlge-domain dials']
        stats['sinlge-domain avg. turns'] = single_samples/stats['sinlge-domain dials']
        stats['multi-domain avg. turns'] = multi_samples/stats['multi-domain dials']
        stats['sinlge-domain samples'] = single_samples
        stats['multi-domain samples'] = multi_samples
        stats['total samples'] = single_samples + multi_samples
        assert sum(delta_c_cnt['single'].values()) == single_samples
        assert sum(delta_c_cnt['multi'].values()) == multi_samples
        single_delta_c_g_1 = single_samples-(delta_c_cnt['single'][0]+delta_c_cnt['single'][1])
        multi_delta_c_g_1 = multi_samples-(delta_c_cnt['multi'][0]+delta_c_cnt['multi'][1])
        stats['sinlge-domain delta_c>1'] = single_delta_c_g_1/single_samples
        stats['multi-domain delta_c>1'] = multi_delta_c_g_1/multi_samples
        stats['sinlge-domain cross-domain delta_c>1'] = cross_domain_cnt['single'][True]/single_delta_c_g_1
        stats['multi-domain cross-domain delta_c>1'] = cross_domain_cnt['multi'][True]/multi_delta_c_g_1
        

        if len(groups) > 1:
            sub_datadir = os.path.join(data_dir, f'group{group_idx}')
            os.makedirs(sub_datadir, exist_ok=True)
        else:
            sub_datadir = data_dir
        full_state = {}
        for data_type, data in zip(['single_domain', 'multi_domain'], [single_domain_data, multi_domain_data]):
            file_name = os.path.join(sub_datadir, f"{data_type}.json")
            with open(file_name, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            for dial in data:
                for turn in dial['turns']:
                    if 'state' in turn:
                        for domain in turn['state']:
                            full_state.setdefault(domain, {})
                            for slot in turn['state'][domain]:
                                full_state[domain][slot] = ''
        with open(os.path.join(sub_datadir, 'full_state.json'), "w", encoding='utf-8') as f:
            json.dump(full_state, f, indent=2)
        
        if group_idx == 0:
            train_data, validation_data, test_data = split_data(single_domain)
            for data, data_split in zip([train_data, validation_data, test_data], ['train', 'validation', 'test']):
                output_filename = os.path.join(sub_datadir, f'{data_split}_single_domain.json')
                stats[f'{data_split} sinlge-domain dials'] = len(data)
                with open(output_filename, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            train_data, validation_data, test_data = split_data(multi_domain, train_ratio=0.4, val_ratio=0.1)
            for data, data_split in zip([train_data, validation_data, test_data], ['train', 'validation', 'test']):
                output_filename = os.path.join(sub_datadir, f'{data_split}_multi_domain.json')
                stats[f'{data_split} multi-domain dials'] = len(data)
                with open(output_filename, "w", encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

            slot_pairs = get_cross_domain_slot_pairs(multi_domain_data, full_state)
            output_filename = os.path.join(sub_datadir, "multi_domain_slot_pairs.json")
            with open(output_filename, "w", encoding='utf-8') as f:
                json.dump(slot_pairs, f, indent=2)
        else:
            src_data_dir = os.path.join(data_dir, f'group0')
            for data_type in ['single_domain', 'multi_domain']:
                for data_split in ['train', 'validation', 'test']:
                    group0_data = json.load(open(os.path.join(src_data_dir, f'{data_split}_{data_type}.json')))
                    filtered_data = [dial for dial in group0_data if all([domain in domain_group for domain in dial['domains']])]
                    output_filename = os.path.join(sub_datadir, f'{data_split}_{data_type}.json')
                    stats[f'{data_split} {data_type.replace("_", "-")} dials'] = len(filtered_data)
                    with open(output_filename, "w", encoding='utf-8') as f:
                        json.dump(filtered_data, f, indent=2)

            slot_pairs = get_cross_domain_slot_pairs(multi_domain_data, full_state)
            output_filename = os.path.join(sub_datadir, "multi_domain_slot_pairs.json")
            with open(output_filename, "w", encoding='utf-8') as f:
                json.dump(slot_pairs, f, indent=2)

        group_stats.append(list(stats.items()))


    table = []
    for i in range(len(group_stats[0])):
        item = {}
        for group_idx, stats in enumerate(group_stats):
            if i < len(stats):
                item[f'group{group_idx} item'] =  stats[i][0]
                item[f'group{group_idx} number'] =  stats[i][1]
        table.append(item)

    res = tabulate(table, headers='keys', tablefmt='github')
    with open(f'{data_dir}/data_stat.md', 'w', encoding='utf-8') as f:
        print(res, file=f)
        
    return data_by_split


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="split data by single domain or multi-domain")
    parser.add_argument('--datasets', '-d', metavar='dataset_name', nargs='*', help='names of unified datasets')
    args = parser.parse_args()
    print(args)
    for dataset_name in tqdm(args.datasets, desc='datasets'):
        data_dir = os.path.join('data', dataset_name)
        data_by_split = create_dst_data(dataset_name, data_dir, args)
