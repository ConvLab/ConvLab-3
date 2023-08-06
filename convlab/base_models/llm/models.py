import json
from copy import deepcopy
from convlab.base_models.llm.base import LLM
from convlab.dialog_agent import Agent
from convlab.nlu import NLU
from convlab.dst import DST
from convlab.nlg import NLG
from convlab.util.unified_datasets_util import load_dataset, load_ontology

class LLM_US(Agent):
    DEFAULT_SYSTEM_INSTRUCTION = " ".join([
        "Imagine you are a user chatting with a helpful assistant to achieve a goal.",
        "You should chat according to the given goal faithfully and naturally.",
        "You should not generate all the information in the goal at once.",
        "You should generate short, precise, and informative response (less than 50 tokens), corresponding to only one or two items in the goal.",
        "You should not generate information not presented in the goal.",
        "If and only if you achieve your goal, express your thanks and generate **\"[END]\"** token.",
        "If you think the assistant can not help you or the conversation falls into a infinite loop, generate **\"[STOP]\"** token."
    ])
    default_reward_func = lambda goal, dialog, completed: 40 if completed else -20

    def __init__(self, api_type, model_name_or_path, system_instruction=DEFAULT_SYSTEM_INSTRUCTION, reward_func=default_reward_func, generation_kwargs=None):
        self.system_instruction = system_instruction
        self.model = LLM(api_type, model_name_or_path, system_instruction, generation_kwargs)
        self.reward_func = reward_func
    
    def reset(self):
        self.model.clear_chat_history()
        self.is_terminated = False
        self.reward = None

    def init_session(self, goal, example_dialog:str=None):
        self.goal = goal
        goal_description = '.\n'.join(['* '+item for item in self.goal['description'].split('. ')])
        system_instruction = f"Goal:\n{goal_description}"
        system_instruction += f"\n\n{self.system_instruction}"
        if example_dialog is not None:
            system_instruction += f"\n\nExample dialog:\n{example_dialog}"
        self.model.set_system_instruction(system_instruction)
        # print(self.model.system_instruction)
        self.reset()

    def is_terminated(self):
        return self.is_terminated
    
    def response(self, message):
        if self.is_terminated:
            return None
        response = self.model.chat(message)
        if "[END]" in response:
            self.reward = self.reward_func(self.goal, self.model.messages, True)
            self.is_terminated = True
        elif "[STOP]" in response:
            self.reward = self.reward_func(self.goal, self.model.messages, False)
            self.is_terminated = True
        return response

    def get_reward(self):
        return self.reward


class LLM_RG(Agent):
    DEFAULT_SYSTEM_INSTRUCTION = " ".join([
        "Imagine you are a helpful assistant that can help the user to complete their task.",
        "You should generate short, precise, and informative response (less than 50 tokens), providing only necessary information.",
        # "You should not generate emoji or special symbols.",
    ])
    def __init__(self, api_type, model_name_or_path, system_instruction=DEFAULT_SYSTEM_INSTRUCTION, generation_kwargs=None):
        self.system_instruction = system_instruction
        self.model = LLM(api_type, model_name_or_path, system_instruction, generation_kwargs)
    
    def reset(self):
        self.model.clear_chat_history()

    def init_session(self, example_dialog:str=None):
        system_instruction = f"{self.system_instruction}"
        if example_dialog is not None:
            system_instruction += f"\n\nExample dialog:\n{example_dialog}"
        self.model.set_system_instruction(system_instruction)
        # print(self.model.system_instruction)
        self.reset()
    
    def response(self, message):
        response = self.model.chat(message)
        return response


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


class LLM_NLU(NLU):
    def __init__(self, dataset_name, api_type, model_name_or_path, speaker, example_dialogs, generation_kwargs=None):
        assert speaker in ['user', 'system']
        self.speaker = speaker
        self.opponent = 'system' if speaker == 'user' else 'user'
        self.ontology = load_ontology(dataset_name)
        self.system_instruction = self.format_system_instruction(self.ontology, example_dialogs)
        print(self.system_instruction)
        self.model = LLM(api_type, model_name_or_path, self.system_instruction, generation_kwargs)

    def format_system_instruction(self, ontology, example_dialogs):
        intents = {intent: ontology['intents'][intent]['description'] for intent in ontology['intents']}
        # domains = {domain: '' for domain in ontology['domains']}
        slots = {domain: {
                    slot: ontology['domains'][domain]['slots'][slot]['description'] 
                    for slot in ontology['domains'][domain]['slots']
                } for domain in ontology['domains']}
        
        # categorical_slot_values = {domain: {
        #                             slot: ontology['domains'][domain]['slots'][slot]['possible_values']
        #                             for slot in ontology['domains'][domain]['slots'] if ontology['domains'][domain]['slots'][slot]['is_categorical']
        #                         } for domain in ontology['domains']}
        
        example = ''
        for example_dialog in example_dialogs:
            for i, turn in enumerate(example_dialog['turns']):
                if turn['speaker'] == self.speaker:
                    if i > 0:
                        example += example_dialog['turns'][i-1]['speaker']+': '+example_dialog['turns'][i-1]['utterance']+'\n'
                    example += turn['speaker']+': '+turn['utterance']+'\n'
                    das = []
                    for da_type in turn['dialogue_acts']:
                        for da in turn['dialogue_acts'][da_type]:
                            intent, domain, slot, value = da.get('intent'), da.get('domain'), da.get('slot', ''), da.get('value', '')
                            das.append((intent, domain, slot, value))
                    example += '<DA>'+json.dumps(das)+'</DA>'+'\n\n'
        
        system_instruction = "\n\n".join([
            """You are an excellent dialogue acts parser. Dialogue acts are used to represent the intention of the speaker. Dialogue acts are a list of tuples, each tuple is in the form of (intent, domain, slot, value). The "intent", "domain", "slot" are defines as follows:""",
            '"intents": '+json.dumps(intents, indent=4),
            # '"domains": '+json.dumps(domains, indent=4),
            '"domain2slots": '+json.dumps(slots, indent=4),
            """Here are example dialogue acts:""",
            example,
            """Now consider the following dialogue. Please generate the dialogue acts of the last utterance of {}. Start with <DA> token and end with </DA> token. Example: "<DA>[["inform", "hotel", "name": "abc"]]</DA>". Do not generate intents, domains, slots that are not defined above.""".format(self.speaker)
        ])
                
        return system_instruction

    
    def predict(self, utterance, context=list()):
        prompt = ""
        for i, turn in enumerate(context[::-1][:1]):
            # only the last utterance of the opponent is used
            if i % 2 == 0:
                prompt = self.opponent+': '+turn+'\n' + prompt
            else:
                prompt = self.speaker+': '+turn+'\n' + prompt
        prompt += self.speaker+': '+utterance+'\n'
        # print('='*50)
        # print('prompt')
        # print(prompt)
        response = self.model.chat(prompt)
        self.model.clear_chat_history()
        # print('response')
        # print(response)
        # print('='*50)
        dialogue_acts = self.normalize_response_to_dialogue_acts(response)
        return dialogue_acts
    
    def normalize_response_to_dialogue_acts(self, response):
        start_token, end_token = "<DA>", "</DA>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return {}
        response = response[start_idx+len(start_token):end_idx].strip()
        if response == "":
            return {}
        try:
            dialogue_acts = json.loads(response)
        except json.decoder.JSONDecodeError:
            # print('JSONDecodeError')
            # print('*'*30)
            # print([response])
            # print('*'*30)
            return {}
        return dialogue_acts
    

class LLM_NLG(NLG):
    def __init__(self, dataset_name, api_type, model_name_or_path, speaker, example_dialogs, generation_kwargs=None):
        assert speaker in ['user', 'system']
        self.speaker = speaker
        self.opponent = 'system' if speaker == 'user' else 'user'
        self.ontology = load_ontology(dataset_name)
        self.system_instruction = self.format_system_instruction(self.ontology, example_dialogs)
        # print(self.system_instruction)
        self.model = LLM(api_type, model_name_or_path, self.system_instruction, generation_kwargs)

    def format_system_instruction(self, ontology, example_dialogs):
        intents = {intent: ontology['intents'][intent]['description'] for intent in ontology['intents']}
        # domains = {domain: '' for domain in ontology['domains']}
        slots = {domain: {
                    slot: ontology['domains'][domain]['slots'][slot]['description'] 
                    for slot in ontology['domains'][domain]['slots']
                } for domain in ontology['domains']}
        
        # categorical_slot_values = {domain: {
        #                             slot: ontology['domains'][domain]['slots'][slot]['possible_values']
        #                             for slot in ontology['domains'][domain]['slots'] if ontology['domains'][domain]['slots'][slot]['is_categorical']
        #                         } for domain in ontology['domains']}
        
        example = ''
        for example_dialog in example_dialogs:
            for i, turn in enumerate(example_dialog['turns']):
                if turn['speaker'] == self.speaker:
                    if i > 0:
                        example += example_dialog['turns'][i-1]['speaker']+': '+example_dialog['turns'][i-1]['utterance']+'\n'
                    das = []
                    for da_type in turn['dialogue_acts']:
                        for da in turn['dialogue_acts'][da_type]:
                            intent, domain, slot, value = da.get('intent'), da.get('domain'), da.get('slot', ''), da.get('value', '')
                            das.append((intent, domain, slot, value))
                    example += '<DA>'+json.dumps(das)+'</DA>'+'\n'
                    example += turn['speaker']+': '+'<UTT>'+turn['utterance']+'</UTT>'+'\n\n'
        
        system_instruction = "\n\n".join([
            """You are an excellent writing machine. You can generate fluent and precise natural language according to the given dialogue acts. Dialogue acts are a list of tuples, each tuple is in the form of (intent, domain, slot, value). The "intent", "domain", "slot" are defines as follows:""",
            '"intents": '+json.dumps(intents, indent=4),
            '"domain2slots": '+json.dumps(slots, indent=4),
            """Here are some examples:""",
            example,
            """Now consider the following dialogue acts. Please generate an utterance of {} that can express the given dialogue acts precisely. Start with <UTT> token and end with </UTT> token. Example: "<UTT>utterance</UTT>". Do not generate unrelated intents, domains, and slots that are not in the given dialogue acts.""".format(self.speaker)
        ])
                
        return system_instruction
    
    def format_dialogue_acts(self, dialogue_acts):
        das = []
                
        if isinstance(dialogue_acts, dict):
            # da in unified format
            for da_type in dialogue_acts:
                for da in dialogue_acts[da_type]:
                    intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
                    das.append((intent, domain, slot, value))
        elif isinstance(dialogue_acts[0], dict):
            # da without da type
            for da in dialogue_acts:
                intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da.get('value', '')
                das.append((intent, domain, slot, value))
        elif isinstance(dialogue_acts[0], list):
            # da is a list of list (convlab-2 format)
            das = dialogue_acts
        else:
            raise ValueError(f"invalid dialog acts format {dialogue_acts}")
        return das

    def generate(self, dialogue_acts, context=list()):
        das = self.format_dialogue_acts(dialogue_acts)
        prompt = ""
        # # relevant concepts
        # prompt += "Relevant concepts:\n"
        # intents = set([da[0] for da in das])
        # prompt += '"intents": '+json.dumps({intent: self.ontology['intents'][intent]['description'] for intent in self.ontology['intents'] if intent in intents}, indent=4)+'\n\n'
        # slots = {}
        # for da in das:
        #     domain, slot = da[1], da[2]
        #     if domain not in slots:
        #         slots[domain] = {}
        #     if slot not in slots[domain] and slot in self.ontology['domains'][domain]['slots']:
        #         slots[domain][slot] = self.ontology['domains'][domain]['slots'][slot]['description']
        # prompt += '"domain2slots": '+json.dumps(slots, indent=4)+'\n\n'

        prompt += self.opponent+': '+context[-1]+'\n'
        prompt += '<DA>'+json.dumps(das)+'</DA>'+'\n\n'
        # print('='*50)
        # print('prompt')
        # print(prompt)
        response = self.model.chat(prompt)
        self.model.clear_chat_history()
        # print('response')
        # print(response)
        # print('='*100)
        response = self.normalize_response(response)
        return response
    
    def normalize_response(self, response):
        start_token, end_token = "<UTT>", "</UTT>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return {}
        response = response[start_idx+len(start_token):end_idx].strip()
        return response


def test_LLM_US_RG():
    dataset = load_dataset('multiwoz21')
    goal = dataset['validation'][0]['goal']
    example_dialog = '\n'.join([f"{turn['speaker'] if turn['speaker'] == 'user' else 'assistant'}: {turn['utterance']}" for turn in dataset['train'][0]['turns']])
    user_model = LLM_US('huggingface', 'Llama-2-7b-chat-hf')
    user_model.init_session(goal)
    system_model = LLM_RG('huggingface', 'Llama-2-7b-chat-hf')
    system_model.init_session()
    system_msg = "Hello, I am a helpful assistant. How may I help you?"
    print()
    print(system_msg)
    max_turn = 20
    while not user_model.is_terminated and max_turn > 0:
        user_msg = user_model.response(system_msg)
        system_msg = system_model.response(user_msg)
        print()
        print(user_msg)
        print(user_model.is_terminated)
        print()
        print(system_msg)
        max_turn -= 1

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

def test_LLM_NLU():
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
    dataset = load_dataset('multiwoz21')
    example_dialogs = dataset['train'][:3]
    nlu = LLM_NLU('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'user', example_dialogs)
    for text, context in zip(texts, contexts):
        # print(text)
        print(nlu.predict(text, context))
        print('-'*50)

def test_LLM_NLG():
    das = [
        { # da in unified format
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
        [ # da without da type
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
        [ # da is a list of list (convlab-2 format)
            ["reqmore", "general", "", ""]
        ],
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
        "You are welcome.  Is there anything else I can help you with today?",
        "No, I am all set.  Have a nice day.  Bye."],
    ]
    dataset = load_dataset('multiwoz21')
    example_dialogs = dataset['train'][:3]
    nlg = LLM_NLG('multiwoz21', 'huggingface', '/data/zhuqi/pre-trained-models/Llama-2-7b-chat-hf', 'system', example_dialogs)
    for da, context in zip(das, contexts):
        print(da)
        print(nlg.generate(da, context))
        print()

if __name__ == '__main__':
    test_LLM_NLG()
