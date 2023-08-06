import json
from convlab.base_models.llm.base import LLM
from convlab.nlg import NLG
from convlab.util.unified_datasets_util import load_dataset, load_ontology


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
    nlg = LLM_NLG('multiwoz21', 'huggingface', 'Llama-2-7b-chat-hf', 'system', example_dialogs)
    for da, context in zip(das, contexts):
        print(da)
        print(nlg.generate(da, context))
        print()

if __name__ == '__main__':
    test_LLM_NLG()
