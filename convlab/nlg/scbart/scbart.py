import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BartTokenizer, BartForConditionalGeneration

from convlab.nlg.nlg import NLG
from convlab.nlg.scbart.util import act2str
from convlab.util.file_util import cached_path

ACT_PLACEHOLDER = "__sem_act_placeholder__"
CON_PLACEHOLDER = "__conduct_placeholder__"
USER_UTT_PLACEHOLDER = "__user_utt_placeholder__"
PROMPT_TEMPLATE_0 = f"The realisation of dialogue actions {ACT_PLACEHOLDER} in natural language with {CON_PLACEHOLDER} conduct is "
PROMPT_TEMPLATE_1 = f"Given the user request '{USER_UTT_PLACEHOLDER}', the realisation of dialogue actions {ACT_PLACEHOLDER} in natural language with {CON_PLACEHOLDER} conduct is "


class SCBART(NLG):
    def __init__(self, dataset_name='multiwoz21', model_path='/home/shutong/models/scbart-nlprompt-semact-conduct', device='cuda'):
        super(SCBART, self).__init__()
        self.dataset_name = dataset_name
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = BartForConditionalGeneration.from_pretrained(
            model_path).to(self.device)

        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        # model_path = cached_path(model_path)
        # self.model.load_state_dict(torch.load(
        # model_path, map_location=torch.device(self.device))['state_dict'])  # model checkpoint: {'epoch': e, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        self.model.eval()
        self.require_conduct = True

    def save_to_pretrained(model_path, output_dir):
        model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base').to('cuda')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cuda'))['state_dict'])
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    def generate(self, action, conduct='neutral', user_utt=None):
        if isinstance(action, dict):
            # da in unified format
            pass
        elif not action:
            return ""
        elif isinstance(action[0], dict):
            # da without da type
            action = {'categorical': action}
        elif isinstance(action[0], list):
            # da is a list of list (convlab-2 format)
            action = {'categorical': [
                {'intent': da[0], 'domain': da[1], 'slot': da[2], 'value': da[3]} for da in action]}
        else:
            raise ValueError(f"invalid dialog acts format {action}")
        action_str = act2str(action)

        if user_utt == None:
            prompt = PROMPT_TEMPLATE_0.replace(
                ACT_PLACEHOLDER, action_str).replace(
                CON_PLACEHOLDER, conduct)
        else:
            prompt = PROMPT_TEMPLATE_1.replace(
                ACT_PLACEHOLDER, action_str).replace(
                CON_PLACEHOLDER, conduct).replace(
                USER_UTT_PLACEHOLDER, user_utt)
        # print(prompt)
        output = self._inference(prompt)[0]
        return output

    def _inference(self, act_str):
        # print(act_str)
        with torch.no_grad():
            model_input = self.tokenizer.encode_plus(
                act_str,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True)
            inputs = model_input["input_ids"].to(self.device)
            # TODO: implement DDP
            output_ids = self.model.generate(
                inputs,
                num_beams=2,
                min_length=0,
                max_length=128,
                do_sample=True,
                temperature=0.9)
            nlg_outputs = self.tokenizer.batch_decode(
                output_ids.cpu(), skip_special_tokens=True)
        return nlg_outputs
