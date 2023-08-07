from convlab.base_models.llm.base import LLM
from convlab.dialog_agent import Agent
from convlab.util.unified_datasets_util import load_dataset

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


if __name__ == '__main__':
    test_LLM_US_RG()
