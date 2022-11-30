import sys
sys.path.append('../../..')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.nn.parallel import DistributedDataParallel as DDP

from convlab.nlg.nlg import NLG
from util import act2str
from scgpt_special_tokens import *


class SCGPT(NLG):
    def __init__(self, dataset_name, model_path, device='cpu'):
        super(SCGPT, self).__init__()
        self.device = device
        self.model = GPT2LMHeadModel(config=GPT2Config.from_pretrained('gpt2-medium')).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        self.model.load_state_dict(torch.load(model_path))

    def generate(self, action):
        action_str = act2str(action)
        output = self._inference_batch([action_str])[0]
        return output

    def _inference_batch(self, sents):
        with torch.no_grad():
            sents = [sent for sent in sents]
            sent_ids = [self.tokenizer.encode(sent) + [self.tokenizer._convert_token_to_id_with_added_voc('&')] for sent in sents]
            max_len = max([len(sent) for sent in sent_ids])
            sent_ids = [sent + [0] * (max_len - len(sent))  for sent in sent_ids]
            inputs = torch.LongTensor(sent_ids).to(self.device)
            model_to_run = self.model.module if type(self.model) is DDP else self.model
            outputs = model_to_run.generate(inputs, max_length=256, attention_mask=(inputs!=0).float(),
                                            eos_token_id=self.tokenizer.pad_token_id)  # greedy
            outputs = outputs[:, len(inputs[0]):]
            def clean_sentence(sent):
                sent = sent.strip()
                if self.tokenizer.eos_token in sent:
                    sent = sent[:sent.index(self.tokenizer.eos_token)]
                return sent
            output_strs = [clean_sentence(item) for item in outputs]
            return output_strs