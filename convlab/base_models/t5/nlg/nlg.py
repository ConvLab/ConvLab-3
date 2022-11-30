import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from convlab.nlg.nlg import NLG
from convlab.base_models.t5.nlu.serialization import serialize_dialogue_acts
from convlab.util.custom_util import model_downloader


class T5NLG(NLG):
    def __init__(self, speaker, context_window_size, model_name_or_path, device='cuda'):
        assert speaker in ['user', 'system']
        self.speaker = speaker
        self.opponent = 'system' if speaker == 'user' else 'user'
        self.context_window_size = context_window_size
        self.use_context = context_window_size > 0

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.model.eval()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logging.info("T5NLG loaded")

    def generate(self, dialogue_acts, context=list()):
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context = context[-self.context_window_size:]
            utts = context + ['']
        else:
            utts = ['']
        input_seq = '\n'.join([f"{self.opponent if (i % 2) == (len(utts) % 2) else self.speaker}: {utt}" for i, utt in enumerate(utts)])
        dialogue_acts_seq = serialize_dialogue_acts(dialogue_acts)
        input_seq = dialogue_acts_seq + '\n' + input_seq
        # print(input_seq)
        input_seq = self.tokenizer(input_seq, return_tensors="pt").to(self.device)
        # print(input_seq)
        output_seq = self.model.generate(**input_seq, max_length=256)
        # print(output_seq)
        output_seq = self.tokenizer.decode(output_seq[0], skip_special_tokens=True)
        # print(output_seq)
        return output_seq


if __name__ == '__main__':
    das = [
        {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {
            "intent": "request",
            "domain": "taxi",
            "slot": "leave at"
            },
            {
            "intent": "request",
            "domain": "taxi",
            "slot": "arrive by"
            }
        ]
        },
        {
        "categorical": [],
        "non-categorical": [
            {
            "intent": "inform",
            "domain": "taxi",
            "slot": "type",
            "value": "blue honda",
            "start": 38,
            "end": 48
            },
            {
            "intent": "inform",
            "domain": "taxi",
            "slot": "phone",
            "value": "07218068540",
            "start": 67,
            "end": 78
            }
        ],
        "binary": [
            {
            "intent": "book",
            "domain": "taxi",
            "slot": ""
            }
        ]
        },
        {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {
            "intent": "reqmore",
            "domain": "general",
            "slot": ""
            }
        ]
        },
        {
        "categorical": [],
        "non-categorical": [],
        "binary": [
            {
            "intent": "bye",
            "domain": "general",
            "slot": ""
            }
        ]
        }
    ]
    contexts = [
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton."],
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "What time do you want to leave and what time do you want to arrive by?",
        "I want to leave after 17:15."],
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "What time do you want to leave and what time do you want to arrive by?",
        "I want to leave after 17:15.",
        "Booking completed! your taxi will be blue honda Contact number is 07218068540",
        "Thank you for all the help! I appreciate it."],
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "What time do you want to leave and what time do you want to arrive by?",
        "I want to leave after 17:15.",
        "Booking completed! your taxi will be blue honda Contact number is 07218068540",
        "Thank you for all the help! I appreciate it.",
        "You are welcome.  Is there anything else I can help you with today?"
        "No, I am all set.  Have a nice day.  Bye."],
    ]
    nlg = T5NLG(speaker='system', context_window_size=0, model_name_or_path='output/nlg/multiwoz21/system/context_3')
    for da, context in zip(das, contexts):
        print(da)
        print(nlg.generate(da, context))
        print()
