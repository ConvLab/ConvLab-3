import os
import pickle
import torch
import torch.utils.data as data
from copy import deepcopy

from tqdm import tqdm

from convlab.policy.vector.vector_binary import VectorBinary
from convlab.util import load_policy_data, load_dataset
from convlab.util.custom_util import flatten_acts
from convlab.util.multiwoz.state import default_state
from convlab.policy.vector.dataset import ActDataset


class PolicyDataVectorizer:
    
    def __init__(self, dataset_name='multiwoz21', vector=None, dst=None):
        self.dataset_name = dataset_name
        if vector is None:
            self.vector = VectorBinary(dataset_name)
        else:
            self.vector = vector
        self.dst = dst
        self.process_data()

    def process_data(self):
        name = f"{self.dataset_name}_"
        name += f"{type(self.dst).__name__}_" if self.dst is not None else ""
        name += f"{type(self.vector).__name__}"
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset, this can take a while..')
            self._build_data(processed_dir)
        
    def _build_data(self, processed_dir):
        self.data = {}

        os.makedirs(processed_dir, exist_ok=True)
        dataset = load_dataset(self.dataset_name)
        data_split = load_policy_data(dataset, context_window_size=2)

        for split in data_split:
            self.data[split] = []
            raw_data = data_split[split]

            if self.dst is not None:
                self.dst.init_session()

            for data_point in tqdm(raw_data):
                if self.dst is None:
                    state = default_state()

                    state['belief_state'] = data_point['context'][-1]['state']
                    state['user_action'] = flatten_acts(data_point['context'][-1]['dialogue_acts'])
                elif "setsumbt" in str(self.dst):
                    last_system_utt = data_point['context'][-2]['utterance'] if len(data_point['context']) > 1 else ''
                    self.dst.state['history'].append(['sys', last_system_utt])

                    usr_utt = data_point['context'][-1]['utterance']
                    state = deepcopy(self.dst.update(usr_utt))
                    self.dst.state['history'].append(['usr', usr_utt])
                elif "trippy" in str(self.dst):
                    # Get last system acts and text.
                    # System acts are used to fill the inform memory.
                    last_system_acts = []
                    last_system_utt = ''
                    if len(data_point['context']) > 1:
                        last_system_acts = []
                        for act_type in data_point['context'][-2]['dialogue_acts']:
                            for act in data_point['context'][-2]['dialogue_acts'][act_type]:
                                value = ''
                                if 'value' not in act:
                                    if act['intent'] == 'request':
                                        value = '?'
                                    elif act['intent'] == 'inform':
                                        value = 'yes'
                                else:
                                    value = act['value']
                                last_system_acts.append([act['intent'], act['domain'], act['slot'], value])
                        last_system_utt = data_point['context'][-2]['utterance']

                    # Get current user acts and text.
                    # User acts are used for internal evaluation.
                    usr_acts = []
                    for act_type in data_point['context'][-1]['dialogue_acts']:
                        for act in data_point['context'][-1]['dialogue_acts'][act_type]:
                            usr_acts.append([act['intent'], act['domain'], act['slot'], act['value'] if 'value' in act else ''])
                    usr_utt = data_point['context'][-1]['utterance']

                    # Update the state for DST, then update the state via DST.
                    self.dst.state['system_action'] = last_system_acts
                    self.dst.state['user_action'] = usr_acts
                    self.dst.state['history'].append(['sys', last_system_utt])
                    self.dst.state['history'].append(['usr', usr_utt])
                    state = deepcopy(self.dst.update(usr_utt))
                else:
                    raise NameError(f"Tracker: {self.dst} not implemented.")

                last_system_act = data_point['context'][-2]['dialogue_acts'] if len(data_point['context']) > 1 else {}
                state['system_action'] = flatten_acts(last_system_act)
                state['terminated'] = data_point['terminated']
                if self.dst is not None and state['terminated']:
                    self.dst.init_session()
                state['booked'] = data_point['booked']
                dialogue_act = flatten_acts(data_point['dialogue_acts'])

                vectorized_state, mask = self.vector.state_vectorize(state)
                vectorized_action = self.vector.action_vectorize(dialogue_act)
                self.data[split].append({"state": vectorized_state, "action": vectorized_action, "mask": mask,
                                         "terminated": state['terminated']})

            with open(os.path.join(processed_dir, '{}.pkl'.format(split)), 'wb') as f:
                pickle.dump(self.data[split], f)

        print("Data processing done.")

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'validation', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)
                
    def create_dataset(self, part, batchsz):
        states = []
        actions = []
        masks = []
        for item in self.data[part]:
            states.append(torch.Tensor(item['state']))
            actions.append(torch.Tensor(item['action']))
            masks.append(torch.Tensor(item['mask']))
        s = torch.stack(states)
        a = torch.stack(actions)
        m = torch.stack(masks)
        dataset = ActDataset(s, a, m)
        dataloader = data.DataLoader(dataset, batchsz, True)
        return dataloader


if __name__ == '__main__':
    data_loader = PolicyDataVectorizer()
