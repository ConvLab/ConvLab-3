import json
import os
from copy import deepcopy
from zipfile import ZipFile
import importlib
from tabulate import tabulate

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
        },
        "binary_dialogue_acts": {
            [
                {
                    "intent": intent name,
                    "domain": domain name,
                    "slot": slot name,
                    "value": some value
                }
            ]
        }
        "state": {
            domain name: {
                slot name: ""
            }
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

    binary_dialogue_acts = set()
    for bda in ontology['binary_dialogue_acts']:
        assert bda['intent'] is None or bda["intent"] in ontology['intents'], f'ONTOLOGY\tintent undefined intent in binary dialog act: {bda}'
        binary_dialogue_acts.add(tuple(bda.values()))
    ontology['bda_set'] = binary_dialogue_acts

    assert 'state' in ontology, 'ONTOLOGY\tno state'
    for domain_name, domain in ontology['state'].items():
        assert domain_name in ontology['domains']
        for slot_name, value in domain.items():
            assert slot_name in ontology['domains'][domain_name]['slots']
            assert value == "", "should set value in state to \"\""

    # print('description existence:', descriptions, '\n')
    for description, value in descriptions.items():
        if not value:
            print(f'description of {description} is incomplete')
    return ontology


def check_dialogues(name, dialogues, ontology):
    global special_values

    all_id = set()
    splits = ['train', 'validation', 'test']
    da_values = 0
    da_matches = 0
    stat_keys = ['dialogues', 'utterances', 'tokens', 'domains']
    stat = {
        split: {
            key: 0 for key in stat_keys
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
            stat[split] = {key: 0 for key in stat_keys}
        
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

        cur_domains = dialogue['domains']
        assert isinstance(cur_domains, list), f'{dialogue_id}\t`domains` is expected to be list type, '
        assert len(set(cur_domains)) == len(cur_domains), f'{dialogue_id}\trepeated domains'
        cur_stat['domains'] += len(cur_domains)
        cur_domains = set(cur_domains)
        for domain_name in cur_domains:
            assert domain_name in ontology['domains'], f'{dialogue_id}\tundefined current domain: {domain_name}'

        # check domain-slot-value
        # prefix: error prefix
        def check_dsv(domain_name, slot_name, value, categorical=None, prefix=f'{dialogue_id}'):
            assert domain_name in cur_domains, f'{prefix}\t{domain_name} not presented in current domains'
            domain = ontology['domains'][domain_name]
            assert slot_name in domain['slots'], f'{prefix}\t{slot_name} not presented in domain {domain_name} in ontology'
            slot = domain['slots'][slot_name]
            if categorical is None:
                # for state
                categorical = slot['is_categorical']
            else:
                # for dialog act
                assert categorical == slot['is_categorical'], \
                    f'{prefix}\t{domain_name}-{slot_name} is_categorical should be {slot["is_categorical"]} as in ontology'
            if categorical:
                for v in value.split('|'):
                    assert v in special_values or v in slot['possible_values'], \
                        f'{prefix}\t`{v}` not presented in possible values of {domain_name}-{slot_name}: {slot["possible_values"]}'

        def check_da(da, categorical):
            assert da['intent'] in ontology['intents'], f'{dialogue_id}:{turn_id}:da\tundefined intent {da["intent"]}'
            check_dsv(da['domain'], da['slot'], da['value'], categorical, f'{dialogue_id}:{turn_id}:da')
        
        goal = dialogue['goal']
        assert isinstance(goal['description'], str), f'{dialogue_id}\tgoal description {goal["description"]} should be string'
        assert isinstance(goal['constraints'], dict), f'{dialogue_id}\tgoal constraints {goal["constraints"]} should be dict'
        assert isinstance(goal['requirements'], dict), f'{dialogue_id}\tgoal requirements {goal["requirements"]} should be dict'
        for domain_name, domain in goal['constraints'].items():
            for slot_name, value in domain.items():
                check_dsv(domain_name, slot_name, value, prefix=f'{dialogue_id}:goal:constraints')
                assert value != "", f'{dialogue_id}\tshould set non-empty value in goal constraints {goal["constraints"]}'
        for domain_name, domain in goal['requirements'].items():
            for slot_name, value in domain.items():
                check_dsv(domain_name, slot_name, value, prefix=f'{dialogue_id}:goal:requirements')
                assert value == "", f'{dialogue_id}\tshould set empty value in goal requirements {goal["requirements"]}'

        turns = dialogue['turns']
        cur_stat['utterances'] += len(turns)
        assert turns, f'{dialogue_id}\tempty turn'

        # assert turns[0]['speaker'] == 'user', f'{dialogue_id}\tnot start with user role'
        for turn_id, turn in enumerate(turns):
            assert turn['speaker'] in ['user', 'system'], f'{dialogue_id}:{turn_id}\tunknown speaker value: {turn["speaker"]}'
            assert turn_id == turn['utt_idx'], f'{dialogue_id}:{turn_id}\twrong utt_idx'
            if turn_id > 0:
                assert turns[turn_id - 1]['speaker'] != turn['speaker'], f'{dialogue_id}:{turn_id}\tuser and system should speak alternatively'

            utterance = turn['utterance']
            cur_stat['tokens'] += len(utterance.strip().split(' '))

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
                    da_values += 1
                    assert ('start' in da) == ('end' in da), \
                        f'{dialogue_id}:{turn_id}\tstart and end field in da should both present or neither not present'
                    if 'start' in da:
                        value = utterance[da['start']:da['end']]
                        if da['value'].lower() == value.lower():
                            da_matches += 1

            for da in dialogue_acts['binary']:
                assert tuple(da.values()) in ontology['bda_set'], f'{dialogue_id}:{turn_id}\tbinary dialog act {da} not present in ontology'
                # do not check_dsv for binary dialogue acts

            if turn['speaker'] == 'user':
                assert 'db_results' not in turn
                assert 'state' in turn, f"{dialogue_id}:{turn_id}\tstate must present in user's role, but could be empty"
                state = turn['state']
                assert isinstance(state, dict), f'{dialogue_id}:{turn_id}\tstate should be a dict'
                for domain_name, domain in state.items():
                    for slot_name, value in domain.items():
                        check_dsv(domain_name, slot_name, value, prefix=f'{dialogue_id}:{turn_id}:state')

            else:
                assert 'state' not in turn, f"{dialogue_id}:{turn_id}\tstate cannot present in system's role"
                assert 'db_results' in turn
                db_results = turn['db_results']
                assert isinstance(db_results, dict), f'{dialogue_id}:{turn_id}\db_results should be a dict'
                for domain_name, results in db_results.items():
                    assert domain_name in cur_domains, f'{dialogue_id}:{turn_id}:db_results\t{domain_name} not presented in current domains'
                    assert isinstance(results, list)

        # assert turns[-1]['speaker'] == 'user', f'{dialogue_id} dialog must end with user role'

    if da_values:
        print('da values span match rate:    {:.3f}'.format(da_matches * 100 / da_values))

    all_stat = {key: 0 for key in stat_keys}
    for key in stat_keys:
        all_stat[key] = sum(stat[split][key] for split in splits)
    stat['all'] = all_stat

    table = []
    for split in splits + ['all']:
        cur_stat = stat[split]
        if cur_stat['dialogues']:
            cur_stat['avg_utt'] = round(cur_stat['utterances'] / cur_stat['dialogues'], 2)
            cur_stat['avg_tokens'] = round(cur_stat['tokens'] / cur_stat['utterances'], 2)
            cur_stat['avg_domains'] = round(cur_stat.pop('domains') / cur_stat['dialogues'], 2)
        else:
            del stat[split]
        table.append({
            'split':split, 
            '\# dialogues': cur_stat['dialogues'], '\# utterances': cur_stat['utterances'],
            'avg_utt': cur_stat['avg_utt'], 'avg_tokens': cur_stat['avg_tokens'], 'avg_domains': cur_stat['avg_domains']
        })
    
    print(f'domains: {len(ontology["domains"])}')
    print('\n\nCopy-and-paste the following statistics to dataset README.md->Dataset Summary section')
    print(tabulate(table, headers='keys', tablefmt='github'))
    print()


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
            print('')
            if not os.path.isdir(name):
                raise FileNotFoundError(f'dataset {name} not found')

            print(f'checking {name}')
            preprocess_file = os.path.join(f'{name}', 'preprocess.py')
            if not os.path.exists(preprocess_file):
                raise FileNotFoundError(f'no {preprocess_file}')

            if args.preprocess:
                print('pre-processing')

                os.chdir(name)
                preprocess = importlib.import_module(f'{name}.preprocess')
                preprocess.preprocess()
                os.chdir('..')

            data_file = os.path.join(f'{name}', 'data.zip')
            if not os.path.exists(data_file):
                raise FileNotFoundError(f'cannot find {data_file}')

            with ZipFile(data_file) as zipfile:
                print('check ontology')
                with zipfile.open('data/ontology.json', 'r') as f:
                    ontology = json.load(f)
                    check_ontology(ontology)
                
                print('check dialogues')
                with zipfile.open('data/dialogues.json', 'r') as f:
                    dialogues = json.load(f)
                    check_dialogues(name, dialogues, ontology)
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
