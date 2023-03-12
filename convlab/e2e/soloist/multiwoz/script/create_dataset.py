import jsonlines
import copy
import fire

from convlab.util.unified_datasets_util import create_delex_data, load_dataset
from convlab.util import load_e2e_data

def state_to_string(state):
    domain_str = []
    for domain,svs in state.items():
        svs_str = []
        for s,v in svs.items():
            if v != '':
                svs_str.append(f'{s} is {v}')
        svs_str = ' ; '.join(svs_str)
        if svs_str != '':
            domain_str.append(f'{domain} {svs_str}')
    domain_str = ' | '.join(domain_str)
    return domain_str

def context_to_string(context):
    response = ' EOS '.join(i['utterance'].strip() for i in context)
    return response

def delex_function(d,s,v):
    s = s.replace(' ','')
    str_ = f'[{d}_{s}]'
    return str_

def create_dataset(mode='joint'):
    dataset_list = {
        'joint': ['tm1','sgd','multiwoz21'],
        'transfer': ['tm1','sgd'],
        'single': ['multiwoz21']
    }

    examples = []
    for _data in dataset_list[mode]:
        
        dataset = load_dataset(_data)
        dataset, delex_vocab = create_delex_data(dataset, delex_func=delex_function)
        e2e_data = load_e2e_data(dataset, delex_utterance = True)
        
        split_list = ['train','validation','test'] if mode == 'single' else ['train']
        
        for split in split_list:
            data = e2e_data[split]
            for i in data:
                response = i['delex_utterance'].strip()
                context = i['context']
                context = context_to_string(context)
                
                example = {}
                example['Context'] = context
                try:
                    knowledge = state_to_string(i['context'][-1]['state'])
                except Exception:
                    knowledge = ''
                example['Knowledge'] = knowledge
                example['Response'] = 'Agent: ' + response.strip()
                example['Dataset'] = f'{_data}'
                examples.append(copy.copy(example))
            if mode == 'single':
                with jsonlines.open(f'./data/{mode}_{split}.jsonl', "w") as writer:
                    for item in examples:
                        writer.write(item)
                examples = []
    if mode != 'single':  
        with jsonlines.open(f'./data/{mode}_train.jsonl', "w") as writer:
            for item in examples:
                writer.write(item)
            
if __name__ == '__main__':
    fire.Fire(create_dataset)