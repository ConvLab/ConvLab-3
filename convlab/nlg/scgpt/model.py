from torch.utils.data import Dataset
from util import act2str
from scgpt_special_tokens import *
import torch
import numpy as np

class SCGPTDataset(Dataset):
    def __init__(self, data, tokenizer):
        """
        Args:
            data: [[da_str, response], [da_str, response], ...]
            tokenizer: GPT2 Tokenizer
        """
        self.data = []
        length_list = []
        for item in data:
            da, response = item['dialogue_acts'], item['utterance']
            da_tokens = tokenizer.encode(act2str(da))
            response_tokens = tokenizer.encode(response)
            length_list.append(len(da_tokens) + len(response_tokens) + 1)
            self.data.append([da_tokens, response_tokens])
        print(f'max: {np.max(length_list)}, min: {np.min(length_list)}, median: {np.quantile(length_list, 0.5)}, 0.99: {np.quantile(length_list, 0.99)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SGD_TMDataset(Dataset):
    def __init__(self, data, tokenizer):
        """
        Args:
            data: [[da_str, response], [da_str, response], ...]
            tokenizer: GPT2 Tokenizer
        """
        self.data = []
        length_list = []
        for item in data:
            da, response = item['dialogue_acts'], item['utterance']
            da_tokens = tokenizer.encode(act2str(da))
            response_tokens = tokenizer.encode(response)
            length_list.append(len(da_tokens) + len(response_tokens) + 1)
            self.data.append([da_tokens, response_tokens])
        print(f'max: {np.max(length_list)}, min: {np.min(length_list)}, median: {np.quantile(length_list, 0.5)}, 0.99: {np.quantile(length_list, 0.99)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]