import sys
sys.path.append('../../..')

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.parallel import DistributedDataParallel as DDP

from convlab.nlg.nlg import NLG
from util import act2str
from scgpt_special_tokens import *

special_tokens = [START_OF_PRED, END_OF_PRED, SYS_SPEAK, USR_SPEAK]

class SCGPT(NLG):
    def __init__(self, dataset_name, model_path, device='cpu'):
        super(SCGPT, self).__init__()
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN, 'eos_token': END_OF_PRED,
                                           'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(model_path))


    def generate(self, action):
        action_str = act2str(action)
        output = self._inference_batch([action_str])[0]
        return output

    def _inference_batch(self, sents):
        with torch.no_grad():
            sents = [sent + ' ' + START_OF_PRED for sent in sents]
            sent_ids = [self.tokenizer.encode(sent) for sent in sents]
            max_len = max([len(sent) for sent in sent_ids])
            sent_ids = [[self.tokenizer.pad_token_id] * (max_len - len(sent)) + sent  for sent in sent_ids]
            inputs = torch.LongTensor(sent_ids).to(self.device)
            model_to_run = self.model.module if type(self.model) is DDP else self.model
            outputs = model_to_run.generate(inputs, max_length=256,
                                            eos_token_id=self.tokenizer.pad_token_id,
                                            pad_token_id=self.tokenizer.pad_token_id)  # greedy
            # outputs = model_to_run.generate(inputs, num_beams=4, max_length=513, eos_token_id=gpt2_tokenizer.eos_token_id,
            #                                 pad_token_id=gpt2_tokenizer.pad_token_id)  # beam search
            
            output_strs = [self.tokenizer.decode(item) for item in outputs]
            return output_strs