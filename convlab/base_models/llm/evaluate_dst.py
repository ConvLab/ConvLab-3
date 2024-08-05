from copy import deepcopy
import json
from time import sleep

from tqdm import tqdm
import backoff
import openai

from convlab.util import load_dataset, load_dst_data
from convlab.base_models.llm.dst import LLM_DST
from convlab.base_models.llm.label_maps import LABEL_MAPS as INV_LABEL_MAPS

LABEL_MAPS = {val: key for key, vals in INV_LABEL_MAPS.items() for val in vals}


def load_data(dataset_name):
    dataset_dict = load_dataset(dataset_name)
    data = load_dst_data(dataset_dict, data_split='test', speaker='all', split_to_turn=False)['test']

    dataset = list()
    for dialog in tqdm(data):
        dialogue_id = dialog['dialogue_id']
        context = list()
        turn_id = 0
        for turn in dialog['turns']:
            context.append(turn['utterance'])
            if turn['speaker'] == 'user':
                _state = deepcopy(turn['state'])
                for domain in _state:
                    for slot in _state[domain]:
                            _state[domain][slot] = LABEL_MAPS.get(_state[domain][slot].lower(),
                                                                  _state[domain][slot].lower())
                _turn = {'dialogue_id': dialogue_id,
                         'turn_id': str(turn_id),
                         'context': deepcopy(context),
                         'state': _state,
                         'predictions': {'state': None}}
                turn_id += 1
                dataset.append(deepcopy(_turn))

    return dataset


def load_model(dataset_name, api_name, model_name_or_path):
    return LLM_DST(dataset_name, api_name, model_name_or_path)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def predict_state(model, turn):
    if turn['turn_id'] == '0':
        model.init_session()
    model.state['history'] = turn['context']
    model.update()

    state = model.state['belief_state']
    for domain in state:
        for slot in state[domain]:
            value = str(state[domain][slot]).lower()
            state[domain][slot] = LABEL_MAPS.get(value, value)

    return state


def evaluate(model, dataset):
    for turn in tqdm(dataset):
        try:
            state = predict_state(model, turn)
        except:
            sleep(60)
            model.init_session()
            state = predict_state(model, turn)
        turn['predictions']['state'] = state

    return dataset


if __name__ == '__main__':
    dataset = load_data('multiwoz21')

    dst = load_model('multiwoz21', 'openai', 'gpt-3.5-turbo')
    dataset = evaluate(dst, dataset)

    with open('multiwoz21_dst_gpt35.json', 'w') as f:
        json.dump(dataset, f, indent=2)
