import json
from tqdm import tqdm
from convlab.nlg.scbart.scbart import SCBART

nlg = SCBART(
    dataset_name='multiwoz21', # default, dummy argument, reserved for future use
    model_path='/Users/shutong/Projects/EmoLoop/checkpoints/nlg/',  # the path to the model checkpoint
    device='cpu' # default cuda.
)

action = {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'food', 'value': 'chinese'}, {'intent': 'inform', 'domain': 'restaurant', 'slot': 'name', 'value': 'the golden wok'}]}
conduct = 'enthusiastic'
prev_utt = 'I am looking for a cheap restaurant.'

o = nlg.generate(action, conduct, prev_utt)

print(o)

