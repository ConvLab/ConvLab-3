import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from convlab.nlu.nlu import NLU
from convlab.base_models.t5.nlu.serialization import deserialize_dialogue_acts
from convlab.util.custom_util import model_downloader


class T5NLU(NLU):
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
        
        logging.info("T5NLU loaded")

    def predict(self, utterance, context=list()):
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context = context[-self.context_window_size:]
            utts = context + [utterance]
        else:
            utts = [utterance]
        input_seq = '\n'.join([f"{self.opponent if (i % 2) == (len(utts) % 2) else self.speaker}: {utt}" for i, utt in enumerate(utts)])
        # print(input_seq)
        input_seq = self.tokenizer(input_seq, return_tensors="pt").to(self.device)
        # print(input_seq)
        output_seq = self.model.generate(**input_seq, max_length=256)
        # print(output_seq)
        output_seq = self.tokenizer.decode(output_seq[0], skip_special_tokens=True)
        # print(output_seq)
        dialogue_acts = deserialize_dialogue_acts(output_seq.strip())
        return dialogue_acts


if __name__ == '__main__':
    texts = [
        "I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "I want to leave after 17:15.",
        "Thank you for all the help! I appreciate it.",
        "Please find a restaurant called Nusha.",
        "I am not sure of the type of food but could you please check again and see if you can find it? Thank you.",
        "It's not a restaurant, it's an attraction. Nusha."
    ]
    contexts = [
        [],
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "What time do you want to leave and what time do you want to arrive by?"],
        ["I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
        "What time do you want to leave and what time do you want to arrive by?",
        "I want to leave after 17:15.",
        "Booking completed! your taxi will be blue honda Contact number is 07218068540"],
        [],
        ["Please find a restaurant called Nusha.",
        "I don't seem to be finding anything called Nusha.  What type of food does the restaurant serve?"],
        ["Please find a restaurant called Nusha.",
        "I don't seem to be finding anything called Nusha.  What type of food does the restaurant serve?",
        "I am not sure of the type of food but could you please check again and see if you can find it? Thank you.",
        "Could you double check that you've spelled the name correctly? The closest I can find is Nandos."]
    ]
    nlu = T5NLU(speaker='user', context_window_size=3, model_name_or_path='output/nlu/multiwoz21/user/context_3')
    for text, context in zip(texts, contexts):
        print(text)
        print(nlu.predict(text, context))
        print()
