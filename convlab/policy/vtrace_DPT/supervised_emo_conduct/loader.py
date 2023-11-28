import os
import pickle
import torch
import json
import time
import torch.utils.data as data

from convlab.policy.vector.vector_binary import VectorBinary
from convlab.util import load_policy_data, load_dataset
from convlab.util.custom_util import flatten_acts
from convlab.util.multiwoz.state import default_state
from convlab.policy.vector.dataset import ActDatasetKG
from tqdm import tqdm
from copy import deepcopy

emotion_dict = {
    0: "Neutral",
    1: "Fearful",
    2: "Dissatisfied",
    3: "Apologetic",
    4: "Abusive",
    5: "Excited",
    6: "Satisfied"
}

conduct_dict = {0: 'neutral', 1: 'compassionate', 2: 'apologetic', 3: 'enthusiastic', 4: 'appreciative'}

emowoz_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "emowoz_2.0.json")))

mwoz_domains = ['restaurant', 'hotel', 'train', 'taxi', 'attraction']


class PolicyDataVectorizer:

    def __init__(self, dataset_name='multiwoz21', vector=None, percentage=1.0, dialogue_order=0, dst=None):
        self.dataset_name = dataset_name
        self.percentage = percentage
        self.dialogue_order = dialogue_order
        if vector is None:
            self.vector = VectorBinary(dataset_name)
        else:
            self.vector = vector
        self.dst = dst
        self.process_data()

    def process_data(self):

        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f'processed_data/{self.dataset_name}_{type(self.vector).__name__}')
        processed_dir += f"_{type(self.dst).__name__}" if self.dst is not None else ""
        if self.percentage != 1.0:
            processed_dir += f"_{self.percentage}_{self.dialogue_order}"
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset, this can take a while..')
            self._build_data(processed_dir)

    def _build_data(self, processed_dir):
        self.data = {}

        os.makedirs(processed_dir, exist_ok=True)
        dataset = load_dataset(self.dataset_name, dial_ids_order=self.dialogue_order,
                               split2ratio={'train': self.percentage, 'validation': self.percentage,
                                            'test': self.percentage})
        data_split = load_policy_data(dataset, context_window_size=2, emotion=True, utt_idx=True)

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
                elif "emodst" in str(self.dst).lower():
                    last_system_utt = data_point['context'][-2]['utterance'] if len(data_point['context']) > 1 else ''
                    self.dst.state['history'].append(['sys', last_system_utt])
                    usr_utt = data_point['context'][-1]['utterance']
                    state = deepcopy(self.dst.update(usr_utt))
                    self.dst.state['history'].append(['usr', usr_utt])
                elif "setsumbt" in str(self.dst):
                    last_system_utt = data_point['context'][-2]['utterance'] if len(data_point['context']) > 1 else ''
                    self.dst.state['history'].append(['sys', last_system_utt])
                    usr_utt = data_point['context'][-1]['utterance']
                    state = deepcopy(self.dst.update(usr_utt))
                    self.dst.state['history'].append(['usr', usr_utt])
                else:
                    NameError(f"Only SetSUMBT is usable at the moment.")

                user_emotion = emotion_dict[data_point['context'][-1]['emotion'][-1]['emotion']].lower()
                system_conduct = conduct_dict[int(
                    emowoz_data["-".join([data_point["dialogue_id"], str(data_point["utt_idx"])])])]

                last_system_act = data_point['context'][-2]['dialogue_acts'] \
                    if len(data_point['context']) > 1 else {}
                state['system_action'] = flatten_acts(last_system_act)
                state['terminated'] = data_point['terminated']

                if 'emotion' not in state:
                    state['emotion'] = user_emotion
                if 'booked' in data_point:
                    state['booked'] = data_point['booked']
                dialogue_act = flatten_acts(data_point['dialogue_acts'])

                if self.dst is not None and state['terminated']:
                    self.dst.init_session()

                vectorized_state, mask = self.vector.state_vectorize(state)
                vectorized_action = self.vector.action_vectorize(dialogue_act)
                self.data[split].append({"state": self.vector.kg_info, "action": vectorized_action, "mask": mask,
                                         "terminated": state['terminated'], "conduct": system_conduct})

            with open(os.path.join(processed_dir, '{}.pkl'.format(split)), 'wb') as f:
                pickle.dump(self.data[split], f)

        print("Data processing done.")

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'validation', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def is_multiwoz_like(self, item):

        state = item['state']
        is_like = False
        for node in state:
            domain = node['domain'].lower()
            for mw_domain in mwoz_domains:
                # we check if the mw_domain as a string is contained in the domain of the node
                if mw_domain in domain:
                    is_like = True
                    break
            if is_like:
                break
        return is_like

    def create_dataset(self, part, batchsz, policy, multiwoz_like=False):
        print('Start creating {} dataset'.format(part))
        time_now = time.time()

        root_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root_dir, "data", self.dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, part)
        file_path += f"_{type(self.dst).__name__}" if self.dst is not None else ""
        if multiwoz_like:
            file_path += "mw"

        if self.percentage != 1.0:
            file_path += f"_{self.percentage}_{self.dialogue_order}"

        if os.path.exists(file_path):
            action_batch, a_masks, max_length, small_act_batch, \
            current_domain_mask_batch, non_current_domain_mask_batch, \
            description_batch, value_batch, kg_list = torch.load(file_path)
            print(f"Loaded data from {file_path}")
        else:
            print("Creating data from scratch.")

            action_batch, small_act_batch, \
            current_domain_mask_batch, non_current_domain_mask_batch, \
            description_batch, value_batch = [], [], [], [], [], []
            kg_list = []

            for num, item in tqdm(enumerate(self.data[part])):

                if item['action'].sum() == 0 or len(item['state']) == 0:
                    continue
                if multiwoz_like:
                    if not self.is_multiwoz_like(item):
                        continue
                action_batch.append(torch.Tensor(item['action']))

                kg = [item['state']]
                kg_list.append(item['state'])

                description_idx_list, value_list = policy.get_descriptions_and_values(kg)
                description_batch.append(description_idx_list)
                value_batch.append(value_list)

                current_domains = policy.get_current_domains(kg)
                current_domain_mask = policy.action_embedder.get_current_domain_mask(current_domains[0], current=True)
                non_current_domain_mask = policy.action_embedder.get_current_domain_mask(current_domains[0], current=False)
                current_domain_mask_batch.append(current_domain_mask)
                non_current_domain_mask_batch.append(non_current_domain_mask)

                small_act = policy.action_embedder.real_action_to_small_action_list(torch.Tensor(item['action']))
                small_act.append(policy.action_embedder.small_action_dict[item['conduct']])
                small_act_batch.append(torch.Tensor(small_act))

            print("Creating action masks..")
            a_masks, max_length = policy.get_action_masks(action_batch, conduct_mask=True)
            action_batch = torch.stack(action_batch)
            current_domain_mask_batch = torch.stack(current_domain_mask_batch)
            non_current_domain_mask_batch = torch.stack(non_current_domain_mask_batch)

            print(f"Finished data set, time spent: {time.time() - time_now}")

            torch.save([action_batch, a_masks, max_length, small_act_batch,
                        current_domain_mask_batch, non_current_domain_mask_batch,
                        description_batch, value_batch, kg_list], file_path)

        dataset = ActDatasetKG(action_batch, a_masks, current_domain_mask_batch, non_current_domain_mask_batch)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print("NUMBER OF EXAMPLES:", len(current_domain_mask_batch))
        return dataloader, max_length, small_act_batch, description_batch, value_batch, kg_list


if __name__ == '__main__':
    data_loader = PolicyDataVectorizer()
