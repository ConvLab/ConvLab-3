import json
from convlab.base_models.llm.base import LLM
from convlab.nlu import NLU
from convlab.util.unified_datasets_util import load_dataset, load_ontology


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


if __name__ == '__main__':
    test_LLM_NLU()
