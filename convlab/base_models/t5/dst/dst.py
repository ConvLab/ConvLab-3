import logging
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from convlab.dst.dst import DST
from convlab.base_models.t5.dst.serialization import deserialize_dialogue_state
from convlab.util import load_ontology


class T5DST(DST):
    def __init__(self, dataset_name, speaker, context_window_size, model_name_or_path, device='cuda'):
        assert speaker in ['user', 'system']
        assert context_window_size > 0
        self.ontology = load_ontology(dataset_name)
        self.speaker = speaker
        self.opponent = 'system' if speaker == 'user' else 'user'
        self.context_window_size = context_window_size
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.model.eval()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logging.info("T5DST loaded")

    def update(self, user_action=None):
        if self.state['history'][0][1] == 'null':
            # skip first dummy turn
            context = self.state['history'][1:]
        else:
            context = self.state['history']
        if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
            context = [item[1] for item in context]
        context = context[-self.context_window_size:]
        input_seq = '\n'.join([f"{self.opponent if (i % 2) == (len(context) % 2) else self.speaker}: {utt}" for i, utt in enumerate(context)])
        # print(input_seq)
        input_seq = self.tokenizer(input_seq, return_tensors="pt").to(self.device)
        # print(input_seq)
        output_seq = self.model.generate(**input_seq, max_length=256)
        # print(output_seq)
        output_seq = self.tokenizer.decode(output_seq[0], skip_special_tokens=True)
        # print(output_seq)
        state = deserialize_dialogue_state(output_seq.strip())
        self.state['belief_state'] = state
        return self.state
    
    def init_session(self):
        self.state = dict()
        self.state['belief_state'] = deepcopy(self.ontology['state'])
        self.state['booked'] = dict()
        self.state['history'] = []
        self.state['system_action'] = []
        self.state['user_action'] = []
        self.state['terminated'] = False


if __name__ == '__main__':
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
        "You are welcome.  Is there anything else I can help you with today?",
        "No, I am all set.  Have a nice day.  Bye."],
    ]
    dst = T5DST('multiwoz21', speaker='user', context_window_size=100, model_name_or_path='ConvLab/t5-small-dst-multiwoz21')
    dst.init_session()
    for context in contexts:
        dst.state['history'] = context
        print(dst.update())
        print()
