import os
import json
import pickle
import torch
import torch.utils.data as data
from convlab2.policy.vector.dataset import ActDataset
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker

PAD = '<pad>'
UNK = '<unk>'
USR = 'YOU:'
SYS = 'THEM:'
BOD = '<d>'
EOD = '</d>'
BOS = '<s>'
EOS = '<eos>'
SEL = '<selection>'
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"
SPECIAL_TOKENS = [PAD, UNK, USR, SYS, BOS, BOD, EOS, EOD]
STOP_TOKENS = [EOS, SEL]
DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]


class DatasetDataloader(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        load data from file, according to what is need
        :param args:
        :param kwargs:
        :return: data
        """
        pass


class MultiWOZDataloader(DatasetDataloader):
    def __init__(self):
        super(MultiWOZDataloader, self).__init__()

    def load_data(self,
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../data/multiwoz')),
                  data_key='all',
                  role='all',
                  utterance=False,
                  dialog_act=False,
                  context=False,
                  context_window_size=0,
                  context_dialog_act=False,
                  belief_state=False,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  ontology=False,
                  session_id=False,
                  span_info=False,
                  terminated=False,
                  goal=False
                  ):


        assert role in ['sys', 'usr', 'all']
        info_list = ['utterance', 'dialog_act', 'context', 'context_dialog_act', 'belief_state',
                                       'last_opponent_utterance', 'last_self_utterance', 'session_id', 'span_info',
                                       'terminated', 'goal']
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role}
        if data_key == 'all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            #data = read_zipped_json(os.path.join(data_dir, '{}.json.zip'.format(data_key)), '{}.json'.format(data_key))
            with open(os.path.join(data_dir, f'{data_key}.json')) as f:
                data = json.load(f)

            print('loaded {}, size {}'.format(data_key, len(data)))

            for x in info_list:
                self.data[data_key][x] = []
            self.data[data_key]["dialogs"] = []

            for sess_id, sess in data.items():

                dialog = {"user": [], "system": [], 'sys_acts': []}

                for i, turn in enumerate(sess['log']):
                    text = turn['text']
                    if i % 2 == 0:
                        self.data[data_key]['utterance'].append(text)
                        dialog['user'].append(text)
                    else:
                        self.data[data_key]['last_self_utterance'].append(text)
                        dialog['system'].append(text)
                        dialog['sys_acts'].append(turn['dialog_act'])

                    self.data[data_key]['session_id'].append(sess_id)
                    self.data[data_key]['terminated'].append(i + 1 >= len(sess['log']))

                self.data[data_key]['dialogs'].append(dialog)

        if ontology:
            ontology_path = os.path.join(data_dir, 'ontology.json')
            self.data['ontology'] = json.load(open(ontology_path))

        return self.data


class UtteranceDataLoaderVRNN:

    def __init__(self, use_confidence_scores=False, use_entropy=False, use_mutual_info=False, use_action_masking=False,
                tracker_path=None):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_utterance_data')
        self.use_confidence_scores = use_confidence_scores
        self.use_entropy = use_entropy
        self.use_mutual_info = use_mutual_info

        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file, use_entropy=self.use_entropy,
                                    use_mutual_info = self.use_mutual_info,
                                    use_confidence_scores=use_confidence_scores)
        self.vector.use_mask = use_action_masking
        print("Action masking activated.")

        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            self.dst = SetSUMBTTracker(model_type="roberta", model_path=tracker_path, get_confidence_scores=use_confidence_scores,
                                        return_entropy=self.use_entropy, return_mutual_info=self.use_mutual_info)
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir)

    def _build_data(self, root_dir, processed_dir):
        self.data = {}
        data_loader = MultiWOZDataloader()
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            raw_data = data_loader.load_data(data_key=part, role='sys')[part]
            dialogs = raw_data['dialogs']

            counter = 0
            for dialog in tqdm(dialogs):
                counter += 1
                self.dst.init_session()

                # if counter % 20 == 0:
                #     print("Dialogues processed: ", counter)

                user_utterances = dialog['user']
                system_utterances = dialog['system']
                system_acts = dialog['sys_acts']

                for user_utt, system_utt, act in zip(user_utterances, system_utterances, system_acts):
                    # Predict dialog state and user actions
                    state = self.dst.update(user_utt)
                    state, action_mask = self.vector.state_vectorize(state)
                    # Vectorize target action set
                    act = [k.split('-', 1)[::-1] + i[0] for k, i in act.items()]
                    action = self.vector.action_vectorize(act)
                    action_mask[action == 1] = 0.0
                    self.data[part].append([state, action_mask, action])
                    # Update history and state with system acts
                    self.dst.state['history'].append(['usr', user_utt])
                    self.dst.state['history'].append(['sys', system_utt])
                    self.dst.state['system_action'] = act

        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz):
        print('Start creating {} dataset'.format(part))
        s = []
        m = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            m.append(torch.Tensor(item[1]))
            a.append(torch.Tensor(item[2]))
        s = torch.stack(s)
        m = torch.stack(m)
        a = torch.stack(a)
        dataset = ActDataset(s, m, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader


if __name__ == '__main__':
    loader = UtteranceDataLoaderVRNN()
    train_data = loader.data['train']

    dialog = train_data[0]
    print("State: ", dialog[0])
    print("Acts: ", dialog[1])
