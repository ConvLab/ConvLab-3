from torch.utils.data import Dataset
from util import act2str
from scgpt_special_tokens import *
import torch

class SCGPTDataset(Dataset):
    def __init__(self, data, tokenizer):
        """
        Args:
            data: [[da_str, response], [da_str, response], ...]
            tokenizer: GPT2 Tokenizer
        """
        self.data = []
        for item in data:
            da, response = item['dialogue_acts'], item['utterance']
            da_tokens = tokenizer.encode(act2str(da))
            response_tokens = tokenizer.encode(response)
            self.data.append([da_tokens, response_tokens])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]