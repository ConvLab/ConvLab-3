import json
import os
from copy import deepcopy
from zipfile import ZipFile
import importlib
from tabulate import tabulate
import random
import string

special_values = ['', 'dontcare', None, '?']


def check_ontology(ontology):
    """
    ontology: {
        "domains": {
            domain name: {
                "description": domain description,
                "slots": {
                    slot name: {
                        "description": slot description,
                        "is_categorical": is_categorical,
                        "possible_values": [possible_values...], not empty if is_categorical
                    }
                }
            }
        },
        "intents": {
            intent name: {
                "description": intent description
            }
        }
        "state": {
            domain name: {
                slot name: ""
            }
        },
        "dialogue_acts": {
            "categorical": [
                "{'user': True/False, 'system': True/False, 'intent': intent, 'domain': domain, 'slot': slot}",
            ],
            "non-categorical": {},
            "binary": {}
        }
    }
    """
    global special_values
    
    # record issues in ontology
    descriptions = {
        # if each domain has a description
        "domains": True,
        "slots": True,
        "intents": True,
    }
    for domain_name, domain in ontology['domains'].items():
        if not domain['description']:
            descriptions["domains"] = False
        for slot_name, slot in domain["slots"].items():
            if not slot["description"]:
                descriptions["slots"] = False
            if slot["is_categorical"]:
                assert slot["possible_values"]
                slot['possible_values'] = list(map(str.lower, slot['possible_values']))
                for value in special_values:
                    assert value not in slot['possible_values'], f'ONTOLOGY\tspecial value `{value}` should not present in possible values'

    for intent_name, intent in ontology["intents"].items():
        if not intent["description"]:
            descriptions["intents"] = False

    assert 'state' in ontology, 'ONTOLOGY\tno state'
    for domain_name, domain in ontology['state'].items():
        assert domain_name in ontology['domains']
        for slot_name, value in domain.items():
            assert slot_name in ontology['domains'][domain_name]['slots']
            assert value == "", "should set value in state to \"\""

    ontology['da_dict'] = {}
    for da_type in ontology['dialogue_acts']:
        ontology['da_dict'][da_type] = {}
        for da_str in ontology['dialogue_acts'][da_type]:
            da = eval(da_str)
            ontology["da_dict"][da_type][(da['intent'], da['domain'], da['slot'])] = {'user': da['user'], 'system': da['system']}

    # print('description existence:', descriptions, '\n')
    for description, value in descriptions.items():
        if not value:
            print(f'description of {description} is incomplete')
    return ontology


def check_dialogues(name, dialogues, ontology):
    global special_values

    all_id = set()
    splits = ['train', 'validation', 'test']
    match_rate = {
        'categorical': {'dialogue act': [0, 0], 'goal': [0, 0], 'state': [0, 0]},
        'noncategorical': {'dialogue act': [0, 0]}
    }
    stat_keys = ['dialogues', 'utterances', 'tokens', 'domains', 
                 'cat slot match(state)', 'cat slot match(goal)', 'cat slot match(dialogue act)',
                 'non-cat slot span(dialogue act)']
    stat = {
        split: {
            key: 0 if 'slot' not in key else [0, 0] for key in stat_keys
        } for split in splits
    }

    # present for both non-categorical or categorical

    for dialogue in dialogues:
        dialogue_id = dialogue['dialogue_id']
        assert isinstance(dialogue_id, str), f'{dialogue_id}\t`dialogue_id` is expected to be str type'

        assert dialogue['dataset'] == name, f'{dialogue_id}\tinconsistent dataset name: {dialogue["dataset"]}'

        split = dialogue['data_split']
        assert isinstance(split, str), f'{dialogue_id}\t`split` is expected to be str type but got {type(split)}'
        if split not in splits:
            splits.append(split)
            stat[split] = {key: 0 if 'slot' not in key else [0, 0] for key in stat_keys}
        
        cur_stat = stat[split]
        cur_stat['dialogues'] += 1
        try:
            prefix, id_split, num = dialogue_id.split('-')
            assert prefix == name and id_split == split
            int(num)    # try converting to int
        except:
            raise Exception(f'{dialogue_id}\twrong dialogue id format: {dialogue_id}')
        assert dialogue_id not in all_id, f'multiple dialogue id: {dialogue_id}'
        all_id.add(dialogue_id)

        if 'domains' in dialogue:
            cur_domains = dialogue['domains']
            assert isinstance(cur_domains, list), f'{dialogue_id}\t`domains` is expected to be list type, '
            # assert len(set(cur_domains)) == len(cur_domains), f'{dialogue_id}\trepeated domains' # allow repeated domains
            cur_stat['domains'] += len(cur_domains)
            cur_domains = set(cur_domains)
            for domain_name in cur_domains:
                assert domain_name in ontology['domains'], f'{dialogue_id}\tundefined current domain: {domain_name}'

        # check domain-slot-value
        # prefix: error prefix
        def check_dsv(domain_name, slot_name, value, anno_type, categorical=None, prefix=f'{dialogue_id}'):
            if anno_type != 'state':
                assert domain_name in cur_domains, f'{prefix}\t{domain_name} not presented in current domains'
            domain = ontology['domains'][domain_name]
            assert slot_name in domain['slots'], f'{prefix}\t{slot_name} not presented in domain {domain_name} in ontology'
            slot = domain['slots'][slot_name]
            if categorical is None:
                # for state and goal
                categorical = slot['is_categorical']
            else:
                # for dialog act
                assert categorical == slot['is_categorical'], \
                    f'{prefix}\t{domain_name}-{slot_name} is_categorical should be {slot["is_categorical"]} as in ontology'
            if categorical and len(value) > 0:
                for v in value.split('|'):
                    stat[split][f'cat slot match({anno_type})'][1] += 1
                    if v in special_values or v.lower() in [s.lower() for s in slot['possible_values']]:
                        stat[split][f'cat slot match({anno_type})'][0] += 1
                    # else:
                    #     print(f'{prefix}\t`{v}` not presented in possible values of {domain_name}-{slot_name}: {slot["possible_values"]}')

        def check_da(da, categorical):
            assert da['intent'] in ontology['intents'], f'{dialogue_id}:{turn_id}:da\tundefined intent {da["intent"]}'
            check_dsv(da['domain'], da['slot'], da['value'], 'dialogue act', categorical, f'{dialogue_id}:{turn_id}:da')
        
        if 'goal' in dialogue:
            goal = dialogue['goal']
            assert isinstance(goal['description'], str), f'{dialogue_id}\tgoal description {goal["description"]} should be string'
            assert isinstance(goal['inform'], dict), f'{dialogue_id}\tgoal inform {goal["inform"]} should be dict'
            assert isinstance(goal['request'], dict), f'{dialogue_id}\tgoal request {goal["request"]} should be dict'
            for domain_name, domain in goal['inform'].items():
                for slot_name, value in domain.items():
                    check_dsv(domain_name, slot_name, value, 'goal', prefix=f'{dialogue_id}:goal:inform')
                    assert value != "", f'{dialogue_id}\tshould set non-empty value in goal inform {goal["inform"]}'
            for domain_name, domain in goal['request'].items():
                for slot_name, value in domain.items():
                    check_dsv(domain_name, slot_name, value, 'goal', prefix=f'{dialogue_id}:goal:request')
                    assert value == "", f'{dialogue_id}\tshould set empty value in goal request {goal["request"]}'

        turns = dialogue['turns']
        cur_stat['utterances'] += len(turns)
        assert turns, f'{dialogue_id}\tempty turn'

        for turn_id, turn in enumerate(turns):
            assert turn['speaker'] in ['user', 'system'], f'{dialogue_id}:{turn_id}\tunknown speaker value: {turn["speaker"]}'
            assert turn_id == turn['utt_idx'], f'{dialogue_id}:{turn_id}\twrong utt_idx'
            if turn_id > 0:
                assert turns[turn_id - 1]['speaker'] != turn['speaker'], f'{dialogue_id}:{turn_id}\tuser and system should speak alternatively'

            utterance = turn['utterance']
            count_zh = 0
            for s in utterance.strip():
                # for Chinese
                if '\u4e00' <= s <= '\u9fff':
                    count_zh += 1
            if count_zh > 0:
                cur_stat['tokens'] += count_zh
            else:
                cur_stat['tokens'] += len(utterance.strip().split(' '))

            if 'dialogue_acts' in turn:
                dialogue_acts = turn['dialogue_acts']
                assert isinstance(dialogue_acts['categorical'], list), f'{dialogue_id}:{turn_id}\tcategorical dialogue_acts should be a list'
                assert isinstance(dialogue_acts['non-categorical'], list), f'{dialogue_id}:{turn_id}\tnon-categorical dialogue_acts should be a list'
                assert isinstance(dialogue_acts['binary'], list), f'{dialogue_id}:{turn_id}\tbinary dialogue_acts should be a list'
                for da in dialogue_acts['categorical']:
                    check_da(da, True)
                for da in dialogue_acts['non-categorical']:
                    check_da(da, False)
                    # values only match after .strip() in some case, it's the issue of pre-processing
                    if da['value'] not in special_values:
                        stat[split][f'non-cat slot span(dialogue act)'][1] += 1
                        assert ('start' in da) == ('end' in da), \
                            f'{dialogue_id}:{turn_id}\tstart and end field in da should both present or neither not present'
                        if 'start' in da:
                            value = utterance[da['start']:da['end']]
                            assert da['value'] == value, f'{dialogue_id}:{turn_id}\tspan({value}) and value{da["value"]} not match' 
                            stat[split][f'non-cat slot span(dialogue act)'][0] += 1

                for da_type in dialogue_acts:
                    for da in dialogue_acts[da_type]:
                        assert ontology['da_dict'][da_type][(da['intent'], da['domain'], da['slot'])][turn['speaker']] == True
                        if da_type == 'binary':
                            assert 'value' not in da, f'{dialogue_id}:{turn_id}\tbinary dialogue act should not have value'

            if turn['speaker'] == 'user':
                assert 'db_results' not in turn
                if 'state' in turn:
                    state = turn['state']
                    assert isinstance(state, dict), f'{dialogue_id}:{turn_id}\tstate should be a dict'
                    for domain_name, domain in state.items():
                        for slot_name, value in domain.items():
                            check_dsv(domain_name, slot_name, value, 'state', prefix=f'{dialogue_id}:{turn_id}:state')

            else:
                assert 'state' not in turn, f"{dialogue_id}:{turn_id}\tstate cannot present in system's role"
                if 'db_results' in turn:
                    db_results = turn['db_results']
                    assert isinstance(db_results, dict), f'{dialogue_id}:{turn_id}\db_results should be a dict'
                    for domain_name, results in db_results.items():
                        # assert domain_name in cur_domains, f'{dialogue_id}:{turn_id}:db_results\t{domain_name} not presented in current domains' # allow query other domains
                        assert isinstance(results, list)

    for _, value_match in match_rate.items():
        for anno_type, (match, total) in value_match.items():
            if total == 0:
                value_match[anno_type] = '-'
            else:
                value_match[anno_type] = '{:.3f}'.format(match*100/total)

    all_stat = {key: 0 if 'slot' not in key else [0, 0] for key in stat_keys}
    for key in stat_keys:
        if 'slot' not in key:
            all_stat[key] = sum(stat[split][key] for split in splits)
        else:
            all_stat[key] = []
            all_stat[key].append(sum(stat[split][key][0] for split in splits))
            all_stat[key].append(sum(stat[split][key][1] for split in splits))
    stat['all'] = all_stat

    table = []
    for split in splits + ['all']:
        cur_stat = stat[split]
        if cur_stat['dialogues']:
            cur_stat['avg_utt'] = round(cur_stat['utterances'] / cur_stat['dialogues'], 2)
            cur_stat['avg_tokens'] = round(cur_stat['tokens'] / cur_stat['utterances'], 2)
            cur_stat['avg_domains'] = round(cur_stat.pop('domains') / cur_stat['dialogues'], 2)
            for key in stat_keys:
                if 'slot' in key:
                    if cur_stat[key][1] == 0:
                        cur_stat[key] = '-'
                    else:
                        cur_stat[key] = round(cur_stat[key][0] * 100 / cur_stat[key][1], 2)
            table.append({
                'split':split, 
                'dialogues': cur_stat['dialogues'], 'utterances': cur_stat['utterances'],
                'avg_utt': cur_stat['avg_utt'], 'avg_tokens': cur_stat['avg_tokens'], 'avg_domains': cur_stat['avg_domains'],
                'cat slot match(state)': cur_stat['cat slot match(state)'], 
                'cat slot match(goal)': cur_stat['cat slot match(goal)'],
                'cat slot match(dialogue act)': cur_stat['cat slot match(dialogue act)'],
                'non-cat slot span(dialogue act)': cur_stat['non-cat slot span(dialogue act)']
            })
        else:
            del stat[split]
    
    return tabulate(table, headers='keys', tablefmt='github')


def create_shuffled_dial_ids(dialogues, rng=random.Random(42), num_orders=10):
    dial_ids = {}
    for i, dialogue in enumerate(dialogues):
        dial_ids.setdefault(dialogue['data_split'], [])
        dial_ids[dialogue['data_split']].append(i)
    
    id_orders = []
    for _ in range(num_orders):
        for data_split in dial_ids:
            rng.shuffle(dial_ids[data_split])
        id_orders.append(deepcopy(dial_ids))
    return id_orders


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="test pre-processed datasets")
    parser.add_argument('datasets', metavar='dataset_name', nargs='*', help='dataset names to be tested')
    parser.add_argument('--all', action='store_true', help='test all datasets')
    parser.add_argument('--no-int', action='store_true', help='not interrupted by exception')
    parser.add_argument('--preprocess', '-p', action='store_true', help='run preprocess automatically')
    args = parser.parse_args()

    if args.all:
        datasets = list(filter(os.path.isdir, os.listdir()))
    else:
        datasets = args.datasets
    if not datasets:
        print('no dataset specified')
        parser.print_help()
        exit(1)

    print('datasets to be tested:', datasets)

    fail = []

    for name in datasets:
        try:
            if not os.path.isdir(name):
                raise FileNotFoundError(f'dataset {name} not found')

            print(f'checking {name}')
            preprocess_file = os.path.join(f'{name}', 'preprocess.py')
            if not os.path.exists(preprocess_file):
                raise FileNotFoundError(f'no {preprocess_file}')

            if args.preprocess:
                print('pre-processing')
                cur_dir = os.getcwd()
                os.chdir(name)
                preprocess = importlib.import_module(f'{name}.preprocess')
                preprocess.preprocess()
                os.chdir(cur_dir)

            data_file = f'{name}/data.zip'
            if not os.path.exists(data_file):
                raise FileNotFoundError(f'cannot find {data_file}')

            with ZipFile(data_file) as zipfile:
                print('check ontology...', end='')
                with zipfile.open('data/ontology.json', 'r') as f:
                    ontology = json.load(f)
                    check_ontology(ontology)
                print('pass')

                print('check dummy data...', end='')
                dummy_data = json.load(open(f'{name}/dummy_data.json'))
                check_dialogues(name, dummy_data, ontology)
                print('pass')
                
                print('check dialogues...', end='')
                with zipfile.open('data/dialogues.json', 'r') as f:
                    dialogues = json.load(f)
                    stat = check_dialogues(name, dialogues, ontology)
                    print('pass')
                    print('creating shuffled_dial_ids')
                    id_orders = create_shuffled_dial_ids(dialogues)
                    with open(os.path.join(name, 'shuffled_dial_ids.json'), 'w', encoding='utf-8') as f:
                        json.dump(id_orders, f, ensure_ascii=False)
                
                print(f'Please copy and paste the statistics in {name}/stat.txt to dataset README.md->Data Splits section\n')
                with open(f'{name}/stat.txt', 'w') as f:
                    print(stat, file=f)
                    print('', file=f)
                    all_domains = list(ontology["domains"].keys())
                    print(f'{len(all_domains)} domains: {all_domains}', file=f)
                    print('- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.', file=f)
                    print('- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.', file=f)

        except Exception as e:
            if args.no_int:
                print(e)
                fail.append(name)
            else:
                raise e

    if not fail:
        print('all datasets passed test')
    else:
        print('failed dataset(s):', fail)
