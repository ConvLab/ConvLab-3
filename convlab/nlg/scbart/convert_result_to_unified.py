import json
import argparse
from copy import deepcopy

import pandas as pd

from convlab.util.unified_datasets_util import load_dataset, load_nlg_data

parser = argparse.ArgumentParser()
parser.add_argument('--predict_result', '-p', type=str, required=True,
                    help='path to the prediction file from train.py')
parser.add_argument('--emowoz2', '-e', type=str, required=True, help='path to the emowoz2.0 conduct labels')
parser.add_argument('--output', '-o', type=str, default='scbart_predict_result_unified.json', help='path to the output file')

args = parser.parse_args()

emowoz2 = json.load(open(args.emowoz2))

df = pd.read_csv(args.predict_result)

predict_result = {}
for i, row in df.iterrows():
    predict_result[row['unique_id']] = row['generation']

dataset_name = 'multiwoz21'
dataset = load_dataset(dataset_name)

nlg_data = load_nlg_data(dataset, 'test', 'system')['test']

nlg_data_copy = deepcopy(nlg_data)

for i, dialog in enumerate(nlg_data):
    dial_id = dialog['dialogue_id']
    turn_id = dialog['utt_idx']

    gen_id = f'{dial_id}-{str(turn_id)}'
    gold_conduct = emowoz2[gen_id]
    scbard_pred_id = f'{gen_id}-{str(gold_conduct)}'
    nlg_data_copy[i]['prediction'] = predict_result[scbard_pred_id]

with open(args.output, 'w') as f:
    json.dump(nlg_data_copy, f, indent=2)
        
