import jsonlines
import json,copy
fidx = open('test.idx.txt','w')

data = json.load(open('data/test.json'))
examples = []
for i in data:
    name = i['file'].lower()   
    history = [] 
    for turn in i['info']:
        history.append(turn['user_orig'])

        bs = turn['BS']
        bs_str = []
        for domain, states in bs.items():
            domain_str = []
            for state in states:
                domain_str.append(f'{state[0]} = {state[1]}')
            domain_str = ' ; '.join(domain_str)
            bs_str.append(domain + ' ' + domain_str)
        bs_str = ' | '.join(bs_str)

        db_str = 'kb '
        db = turn['KB']
        if db == 0:
            db_str += 'zero'
        elif db_str == 1:
            db_str += 'one'
        elif db_str == 2:
            db_str += 'two'
        else:
            db_str += 'more than two'

        act_seq = ' '.join(turn['act'].keys())
        example = {}
        example['Context'] = ' EOS '.join(history[:])
        example['Knowledge'] = ''
        example['Response'] = 'belief : ' + bs_str + ' EOS ' + turn['sys'].strip()

        history.append(turn['sys'].strip())
        examples.append(copy.copy(example))
        fidx.write(name + '\n')

writer =  jsonlines.open('multiwoz_test_e2e.jsonl', mode='w')
for i in examples:    
    writer.write(i)


data = json.load(open('data/val.json'))
examples = []
for i in data:
    name = i['file'].lower()   
    history = [] 
    for turn in i['info']:
        history.append(turn['user_orig'])


        bs = turn['BS']
        bs_str = []
        for domain, states in bs.items():
            domain_str = []
            for state in states:
                domain_str.append(f'{state[0]} = {state[1]}')
            domain_str = ' ; '.join(domain_str)
            bs_str.append(domain + ' ' + domain_str)
        bs_str = ' | '.join(bs_str)

        db_str = 'kb '
        db = turn['KB']
        if db == 0:
            db_str += 'zero'
        elif db_str == 1:
            db_str += 'one'
        elif db_str == 2:
            db_str += 'two'
        else:
            db_str += 'more than two'

        act_seq = ' '.join(turn['act'].keys())
        example = {}
        example['Context'] = ' EOS '.join(history[:])
        example['Knowledge'] = ''
        example['Response'] = 'belief : ' + bs_str + ' EOS ' + turn['sys'].strip()

        history.append(turn['sys'].strip())
        examples.append(copy.copy(example))
        # fidx.write(name + '\n')

writer =  jsonlines.open('multiwoz_valid_e2e.jsonl', mode='w')
for i in examples:    
    writer.write(i)


data = json.load(open('data/train.json'))
examples = []
for i in data:
    name = i['file'].lower()   
    history = [] 
    for turn in i['info']:
        history.append(turn['user_orig'])


        bs = turn['BS']
        bs_str = []
        for domain, states in bs.items():
            domain_str = []
            for state in states:
                domain_str.append(f'{state[0]} = {state[1]}')
            domain_str = ' ; '.join(domain_str)
            bs_str.append(domain + ' ' + domain_str)
        bs_str = ' | '.join(bs_str)

        db_str = 'kb '
        db = turn['KB']
        if db == 0:
            db_str += 'zero'
        elif db_str == 1:
            db_str += 'one'
        elif db_str == 2:
            db_str += 'two'
        else:
            db_str += 'more than two'

        act_seq = ' '.join(turn['act'].keys())
        example = {}
        example['Context'] = ' EOS '.join(history[:])
        example['Knowledge'] = ''
        example['Response'] = 'belief : ' + bs_str + ' EOS ' + turn['sys'].strip()

        history.append(turn['sys'].strip())
        examples.append(copy.copy(example))
        # fidx.write(name + '\n')

writer =  jsonlines.open('multiwoz_train_e2e.jsonl', mode='w')
for i in examples:    
    writer.write(i)
