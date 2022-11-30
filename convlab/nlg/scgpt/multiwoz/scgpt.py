import torch
import numpy as np
import os
import zipfile
from copy import deepcopy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from convlab.nlg.scgpt.utils import tuple2seq
from convlab.nlg.scgpt.decode import set_seed, sample_sequence
from convlab.nlg.nlg import NLG
from convlab.util.file_util import cached_path

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

class SCGPT(NLG):
    
    def __init__(self, model_file=None,
                 use_cuda=True, is_user=False):
        # If no filename is mentioned then set to default
        if not model_file:
            if is_user:
                model_file = 'https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/nlg-gpt-multiwoz.zip'
            else:
                model_file = 'https://zenodo.org/record/5767426/files/neo_scgpt_system.zip'

        # Load from file/url
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isfile(model_file):
            model_file = cached_path(model_file)
        if not os.path.isdir(model_file):
            archive = zipfile.ZipFile(model_file, 'r')
            archive.extractall(model_dir)
            # Get model directory
            model_file = archive.filelist[0].filename.replace('/', '')
            self.model_name_or_path = os.path.join(model_dir, model_file)
        else:
            self.model_name_or_path = model_file
            
        self.length = 50
        self.num_samples = 5
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.top_k = 50
        self.top_p = 0.9
        self.seed = 42
        self.is_user = is_user
        self.stop_token = '<|endoftext|>'
    
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        set_seed(self.seed, torch.cuda.device_count())
    
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)
        self.model = model_class.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
    
        if self.length < 0 and self.model.config.max_position_embeddings > 0:
            self.length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < self.length:
            self.length = self.model.config.max_position_embeddings  # No generation bigger than model size 
        elif self.length < 0:
            self.length = self.MAX_LENGTH  # avoid infinite loop
        
        self.init_session()
    
    def init_session(self):
        self.sess_domains = {'Attraction':False,
            'Hospital':False,
            'Hotel':False,
            'Police':False,
            'Restaurant':False,
            'Taxi':False,
            'Train':False,}
        self.cur_domain = None
        # if not self.is_user:
        #     self.sess_domains['Booking'] = False
                
    def generate(self, meta):

        #some actions in testing data is none
        if not meta:
            return 'No user action'

        meta = deepcopy(meta)
        for list_ in meta:
            domain = list_[1]
            if domain not in ('general', 'Booking'):
                self.cur_domain = domain
        for i, list_ in enumerate(meta):
            list_ = list(list_)
            if list_[1] == 'Booking':
                if self.cur_domain is not None:
                    list_[1] = self.cur_domain
                    meta[i] = list_
                else:
                    print('`cur_domain` is None, but there is `Booking` in dialog action.')
        raw_text = tuple2seq(meta)
        domains = set([item[1] for item in meta])
        for domain in domains:
            if domain not in ('general', 'Booking') and not self.sess_domains[domain]:
                raw_text = raw_text.replace(domain.lower(), domain.lower()+ ' *', 1)
                self.sess_domains[domain] = True
        context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)
        out = sample_sequence(
            model=self.model,
            context=context_tokens,
            num_samples=self.num_samples,
            length=self.length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            device=self.device,
        )
        out = out[:, len(context_tokens):].tolist()
        index = np.random.choice([0,1,2,3],p=[0.4,0.3,0.2,0.1])
        o = out[index]
        text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text.split('& ')[-1]
        text = text[: text.find(self.stop_token) if self.stop_token else None]
    
        return text
