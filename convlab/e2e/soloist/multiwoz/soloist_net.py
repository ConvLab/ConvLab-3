import logging
import torch

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from convlab.e2e.soloist.multiwoz.config import global_config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def cuda_(var):
    return var.cuda() if cfg.cuda and torch.cuda.is_available() else var


def tensor(var):
    return cuda_(torch.tensor(var))

class SOLOIST:

    def __init__(self) -> None:
        
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path,config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        print('model loaded!')

        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

    def generate(self, inputs):

        self.model.eval()
        inputs = self.tokenizer([inputs])
        input_ids = tensor(inputs['input_ids'])
        generated_tokens = self.model.generate(input_ids = input_ids, max_length = cfg.max_length, top_p=cfg.top_p)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return decoded_preds[0]

    
    