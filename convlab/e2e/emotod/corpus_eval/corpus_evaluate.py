import os
import json
from pprint import pprint

import statistics

# pip install git+https://github.com/Tomiinek/MultiWOZ_Evaluation.git@master
from mwzeval.metrics import Evaluator

e = Evaluator(bleu=True, success=True, richness=True)

for sys_name in ['emoloop', 'emoloop_base', 'emoloop_express', 'emoloop_recognise']:
    BLEU, CBE, TRI = [], [], []
    for seed in ['0', '1', '2', '3', '4', '5']:
        pred_f = f'{sys_name}/predictions-{seed}.json'
        # pred_f = 'out.json'
        if not os.path.exists(pred_f):
            continue
        print(f'Processing {pred_f}')
        predictions = json.load(open(pred_f, 'r'))
        results = e.evaluate(predictions)
        # pprint(results)
        # exit()
        
        BLEU.append(results['bleu']['mwz22'])
        CBE.append(results['richness']['cond_entropy'])
        TRI.append(results['richness']['num_trigrams'])

    bleu_mean = statistics.mean(BLEU)
    cbe_mean = statistics.mean(CBE)
    tri_mean = statistics.mean(TRI)

    print(sys_name)
    print(bleu_mean)
    print(cbe_mean)
    print(tri_mean)
