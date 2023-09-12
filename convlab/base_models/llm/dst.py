import json
from copy import deepcopy
from convlab.base_models.llm.base import LLM
from convlab.dst import DST
from convlab.util.unified_datasets_util import load_ontology


class LLM_DST(DST):
    def __init__(self, dataset_name, api_type, model_name_or_path, generation_kwargs=None):
        self.ontology = load_ontology(dataset_name)
        self.system_instruction = self.format_system_instruction(self.ontology)
        # print(self.system_instruction)
        self.model = LLM(api_type, model_name_or_path, self.system_instruction, generation_kwargs)
        self.state_update = []

    def format_system_instruction(self, ontology):
        # From paper "ChatGPT for Zero-shot Dialogue State Tracking: A Solution or an Opportunity?"
        # http://arxiv.org/abs/2306.01386
        state = ontology['state']
        slot_descriptions = deepcopy(ontology['state'])
        categorical_slot_values = deepcopy(ontology['state'])
        
        for domain in state:
            for slot in state[domain]:
                slot_descriptions[domain][slot] = ontology['domains'][domain]['slots'][slot]['description']
                if ontology['domains'][domain]['slots'][slot]['is_categorical']:
                    categorical_slot_values[domain][slot] = ontology['domains'][domain]['slots'][slot]['possible_values']
                else:
                    categorical_slot_values[domain].pop(slot)
            if categorical_slot_values[domain] == {}:
                categorical_slot_values.pop(domain)
        
        system_instruction = "\n\n".join([
            """Consider the following list of concepts , called "slots" provided to you as a json dictionary.""",
            "\"slots\": "+json.dumps(slot_descriptions, indent=4),
            """Some "slots" can only take a value from predefined list:""",
            "\"categorical\": "+json.dumps(categorical_slot_values, indent=4),
            """Now consider the following dialogue between two parties called the "system" and "user". Can you tell me which of the "slots" were updated by the "user" in its latest response to the "system"?""",
            """Present the updates in **JSON** format, start with <JSON> token and end with </JSON> token. Example: "<JSON>{"hotel": {"name": "abc"}}</JSON>". **Do not forget the "}" token**. If no "slots" were updated, return an empty JSON dictionary. If a user does not seem to care about a discussed "slot" fill it with "dontcare"."""
        ])
                
        return system_instruction
    
    def format_turn_prompt(self, user_utterance, system_utterance):
        return '"system": "{}"\n"user": "{}"'.format(system_utterance, user_utterance)

    def normalize_response_to_state_update(self, response):
        start_token, end_token = "<JSON>", "</JSON>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return {}
        response = response[start_idx+len(start_token):end_idx].strip()
        if response == "":
            return {}
        try:
            state_update = json.loads(response)
        except json.decoder.JSONDecodeError:
            # print('JSONDecodeError')
            # print('*'*30)
            # print([response])
            # print('*'*30)
            return {}
        return state_update

    def update(self, user_action=None):
        assert user_action == None
        context = self.state['history']
        assert len(context) > 0
        if type(context[0]) is list:
            assert len(context[0]) > 1
            context = [item[1] for item in context]
        if len(context) % 2 == 0:
            # system/user/system/user
            assert context[0] == ''
        else:
            # first turn: empty system utterance
            context.insert(0, '')
        
        assert len(context)//2 >= len(self.state_update) + 1
        for i in range(len(self.state_update), len(context)//2):
            system_utterance = context[2*i]
            user_utterance = context[2*i+1]
            turn_prompt = self.format_turn_prompt(user_utterance, system_utterance)
            response = self.model.chat(turn_prompt)
            state_update = self.normalize_response_to_state_update(response)
            # print(turn_prompt)
            # print(response)
            # print(state_update)
            # print('---'*50)
            self.state_update.append(state_update)

        self.state['belief_state'] = deepcopy(self.ontology['state'])
        for state_update in self.state_update:
            for domain in state_update:
                if domain not in self.state['belief_state']:
                    continue
                for slot in state_update[domain]:
                    if slot not in self.state['belief_state'][domain]:
                        continue
                    self.state['belief_state'][domain][slot] = state_update[domain][slot]
        return self.state
    
    def init_session(self):
        self.state = dict()
        self.state['belief_state'] = deepcopy(self.ontology['state'])
        self.state['booked'] = dict()
        self.state['history'] = []
        self.state['system_action'] = []
        self.state['user_action'] = []
        self.state['terminated'] = False
        self.state_update = []
        self.model.clear_chat_history()


def test_LLM_DST():
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
    dst = LLM_DST('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf')
    dst.init_session()
    for context in contexts:
        dst.state['history'] = context
        # dst.update()
        print(dst.update())
        print('='*100)


if __name__ == '__main__':
    test_LLM_DST()
