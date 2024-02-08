import json
from pprint import pprint
from copy import deepcopy

from tqdm import tqdm
from convlab.util import load_dataset
from convlab.dst.setsumbt.tracker import SetSUMBTTracker

with open('predictions.json', 'r') as f:
    predictions = json.load(f, indent=4)

### load utterances






# pip install git+https://github.com/Tomiinek/MultiWOZ_Evaluation.git@master
from mwzeval.metrics import Evaluator

e = Evaluator(bleu=True, success=False, richness=False)

results = e.evaluate(predictions)

