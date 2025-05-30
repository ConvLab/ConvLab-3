# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""
import copy
import time
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from peft import LoraConfig, get_peft_model

from convlab.dialog_agent import Agent
from convlab.e2e.emotod.utils import lexcalise, find_substrings, additional_special_tokens_llama
from convlab.util import load_database
from convlab.util.custom_util import NumpyEncoder


EMOTION_PLACEHOLDER = '__emotion_placeholder__'

class EMOLLAMAAgent(Agent):
    def __init__(self,
                 context_size=15,
                 max_output_len=128,
                 model_file='path_to_the_trained_model',
                 base_model_path='path_to_the_base_llama2_model',
                 name='emollama',
                 simple=False,
                 device='cuda:0'):
        super(EMOLLAMAAgent, self).__init__(name=name)

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)#, device_map="auto")
    
        lora_config = LoraConfig(
            r=32, # matrix dim
            lora_alpha=32, # alpha scaling
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            base_model_name_or_path=base_model_path,
            modules_to_save=['lm_head', 'embed_tokens'],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens_llama})
        # adapt embedding layer to the new vocabulary
        base_model.resize_token_embeddings(len(self.tokenizer))
        # model.load_state_dict(f'{model_path}/pytorch_model.bin')

        peft_model = get_peft_model(base_model, lora_config)
        peft_model.load_state_dict(torch.load(f'{model_file}/pytorch_model.bin'), strict=False)
        self.model = peft_model.to(device)

        self.model.eval()

        self.device = self.model.device
        self.eos_token = "<|endofresponse|>" 
        self.dataset_name = 'multiwoz21'
        self.database = load_database(self.dataset_name)

        self.context_size = context_size
        assert self.context_size % 2 == 1
        self.max_output_len = max_output_len
        self.simple = simple

        self.info_dict = {}
        self.init_session()
        self.current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.io_log = []
        self.model_name = name

    
    def init_session(self):
        self.utterance_history = []
        self.user_emotion_history = []
        self.info_dict = {}
    
    def prepare_input(self, usr):
        self.utterance_history.append(usr)

        trunc_utt_hist = copy.deepcopy(self.utterance_history[-self.context_size:])
        trunc_usr_emo_hist = copy.deepcopy(self.user_emotion_history[-(self.context_size//2):])
        
        context = "<|context|>"
        for i, t in enumerate(trunc_utt_hist):
            if i%2 == 0: # user turn
                context += f" <|user|> {t}"
                if not self.simple:
                    if i//2 < len(trunc_usr_emo_hist):
                        context += f" <|useremotion|> {EMOTION_PLACEHOLDER} <|endofuseremotion|>"
                        context = context.replace(EMOTION_PLACEHOLDER, trunc_usr_emo_hist[i//2])
            else:
                context += f" <|system|> {t}"
                
        context += " <|endofcontext|> "

        return context
    
    def predict(self, usr):
        return self.response(usr)

    def response(self, usr):
        """
        Generate agent response given user input.

        Args:
            observation (str):
                The input to the agent.
        Returns:
            response (str):
                The response generated by the agent.
        """
        context = self.prepare_input(usr)

        # self.info_dict['utterance_history'] = copy.deepcopy(self.utterance_history)
        # self.info_dict['user_emotion_history'] = copy.deepcopy(self.user_emotion_history)
        # self.info_dict['input'] = context
        
        encoding = self.tokenizer(context, return_tensors="pt", padding=True).to(self.device)
        
        # while encoding.input_ids.shape[2] > 

        outputs = self.model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False, 
            max_new_tokens=500, 
            eos_token_id=self.tokenizer.convert_tokens_to_ids([self.eos_token])[0],
            no_repeat_ngram_size=10,
            )
        
        full_generation = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        self.io_log.append({'input': context, 'output': full_generation})   
        file_path = f"raw-{self.model_name}-{self.current_time}.json"
        json.dump(self.io_log,
                  open(file_path, 'w'),
                  cls=NumpyEncoder, indent=2)

        # get user emotion and update to user_emotion_history
        if not self.simple:
            user_emotion_str = find_substrings(full_generation, '<|useremotion|>', '<|endofuseremotion|>')
            if not user_emotion_str:
                user_emotion_str = 'neutral'
            else:
                user_emotion_str = user_emotion_str[-1].strip()
            self.user_emotion_history.append(user_emotion_str)

        lexicalised_response = lexcalise(full_generation, self.database).strip()
        # print(full_generation)
        self.utterance_history.append(lexicalised_response)

        self.info_dict['model_output'] = full_generation
        self.info_dict['lexicalised_response'] = lexicalised_response

        return lexicalised_response

if __name__ == '__main__':
    s = EMOLLAMAAgent(
        context_size=15,
        max_output_len=128,
        model_file='path_to_the_trained_model',
        base_model_path='path_to_the_base_llama2_model',
        name='emollama',
        simple=False,
        device='cuda:0'
    )

    user = "I want to find a cheap restaurant in the center"
    system = s.response(user)
    print(user)
    print(system)
    "There are 15 cheap restaurants in the centre . What type of food do you want ?"
    print()

    user = "I would like to have chinese food"
    system = s.response(user)
    "There are 3 cheap chinese restaurants in the centre . Would you like me to make a reservation for you at 1 of them ?"
    print(user)
    print(system)

    user = "Yes, please reserve for two people at 6 pm on monday"
    system = s.response(user)
    print(user)
    print(system)
    "I have booked you at Charlie Chan . The reference number is 00000010 . Is there anything else i can help you with ?"

    