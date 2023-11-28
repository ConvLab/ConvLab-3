import json
from tqdm import tqdm
from convlab.nlg.scbart.scbart import SCBART

conv_log = json.load(open('/home/shutong/emoUS_v2_results_conversation_nlg_langEmoUS-augSCBart_conversation.json', 'r'))
conv = conv_log['conversation']
actions = []
conducts = []
for c in conv:
    for turn in c['log']:
        if turn['role'] == 'sys':
            actions.append(turn['act'])
            conducts.append(turn['conduct'])

nlg = SCBART(
    dataset_name='multiwoz21', # default, dummy argument, reserved for future use
    model_path='/home/shutong/models/scbart-nlprompt-semact-conduct',  # the path to the model checkpoint
    device='cuda' # default cuda.
)

output_f = 'nlg_output.json'
output = []
for a, c in tqdm(zip(actions, conducts)):
    o = nlg.generate(a, c)
    output.append({'act': a, 'conduct': c, 'nlg': o})

json.dump(output, open(output_f, 'w'), indent=4)