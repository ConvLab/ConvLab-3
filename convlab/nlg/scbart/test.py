from convlab.nlg.scbart.scbart import SCBART

ipt = {'categorical': [{'intent': 'inform', 'domain': 'restaurant', 'slot': 'area', 'value': 'north'}], 'non-categorical': [{'intent': 'inform', 'domain': 'hotel', 'slot': 'area', 'value': 'north'}], 'binary': [{'intent': 'request', 'domain': 'hotel', 'slot': 'area'}]}

nlg = SCBART(
    dataset_name='multiwoz21', # default, dummy argument, reserved for future use
    model_path='/home/shutong/models/scbart-nlprompt-semact-conduct-aug-downsample',  # the path to the model checkpoint
    device='cuda' # default cuda.
)

output = nlg.generate(ipt, 'appreciative')

print(output)