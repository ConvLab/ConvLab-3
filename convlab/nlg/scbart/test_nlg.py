import json
from tqdm import tqdm
from convlab.nlg.scbart.scbart import SCBART

model_path = '/home/shutong/models/scbart-nlprompt-semact-conduct-userutt-aug-withdialmage.pt'
model_out_path = '/home/shutong/models/scbart-nlprompt-semact-conduct-userutt-aug-withdialmage'
SCBART.save_to_pretrained(model_path, model_out_path)
exit()
conv_log = json.load(open('/home/shutong/emoUS_v2_results_conversation_nlg_langEmoUS-augSCBart_conversation.json', 'r'))
conv = conv_log['conversation']
actions = []
conducts = []
prev_utts = []
for c in conv:
    for i, turn in enumerate(c['log']):
        if turn['role'] == 'sys':
            actions.append(turn['act'])
            conducts.append(turn['conduct'])
            prev_utts.append(c['log'][i-1]['utt'])

nlg = SCBART(
    dataset_name='multiwoz21', # default, dummy argument, reserved for future use
    model_path='/home/shutong/models/scbart-nlprompt-semact-conduct-userutt-aug',  # the path to the model checkpoint
    device='cuda' # default cuda.
)

output_f = 'nlg_output.json'
output = []
for a, c, p in tqdm(zip(actions, conducts, prev_utts)):
    o = nlg.generate(a, c, p)
    print(a)
    print(c)
    print(p)
    print(o)
    print('-----')
    output.append({'act': a, 'conduct': c, 'nlg': o})
    break

# json.dump(output, open(output_f, 'w'), indent=4)
