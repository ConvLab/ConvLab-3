"""Wrapper for LLMs' API. Need transformers>=4.31.0 to use Llama-2."""
import os
import openai
import torch
from copy import deepcopy
from transformers import pipeline, AutoTokenizer, AutoModel
from litellm import completion as litellm_completion

class LLM:
    def __init__(self, api_type, model_name_or_path, system_instruction=None, generation_kwargs=None):
        """Initialize the LLM wrapper
        api_type: str, should be in API_TYPES
        model_name_or_path: str, the model's id, name, or local path
        system_instruction: str, the system_instruction prompt
        generation_kwargs: dict, kwargs for the generation function
        """
        if api_type == 'openai':
            self.model = OpenAI_API(model_name_or_path)
            
        elif api_type == 'huggingface':
            if 'Llama-2' in model_name_or_path:
                self.model = LLaMa2(model_name_or_path)
            elif 'chatglm2' in model_name_or_path:
                self.model = ChatGLM2(model_name_or_path)
            else:
                self.model = HFModels(model_name_or_path)
        elif api_type == 'litellm':
            self.model = LiteLLM_API(model_name_or_path)
        else:
            raise NotImplementedError
        
        self.system_instruction = self.model.DEFAULT_SYSTEM_INSTRUCTION if system_instruction is None else system_instruction
        self.messages = [{"role": "system", "content": self.system_instruction}]
        self.generation_kwargs = {} if generation_kwargs is None else generation_kwargs
            
    def set_system_instruction(self, system_instruction):
        """Set the system instruction"""
        self.system_instruction = system_instruction
        self.messages[0] = {"role": "system", "content": self.system_instruction}
        
    def chat(self, message, **kwargs):
        """Chat with the LLM"""
        self.messages.append({"role": "user", "content": message})
        messages = deepcopy(self.messages)
        assert len(messages) % 2 == 0
        assert all([m["role"] == "user" for m in messages[1::2]]) and all([m["role"] == "assistant" for m in messages[2::2]])
        response = self.model.chat(messages, **{**self.generation_kwargs, **kwargs})
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.messages = [{"role": "system", "content": self.system_instruction}]
    
    def generate(self, prompt, **kwargs):
        """Generate a response given a prompt"""
        response = self.model.generate(self.system_instruction, prompt, **{**self.generation_kwargs, **kwargs})
        return response

class BaseLLM:
    DEFAULT_SYSTEM_INSTRUCTION = ''

    def __init__(self, model_name_or_path):
        """Initialize the LLM wrapper"""
        raise NotImplementedError

    def chat(self, messages, **kwargs):
        """Chat interface, need to prepare the messages as the format of [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create)"""
        raise NotImplementedError
    
    def generate(self, system_instruction, prompt, **kwargs):
        """Generate interface"""
        raise NotImplementedError
    
class LiteLLM_API(BaseLLM):
    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."
    def __init__(self, model_name_or_path) -> None:
        # make sure you set the corresponding API_KEY environment variable for your model
        self.model_name_or_path = model_name_or_path
    
    def chat(self, messages, **kwargs) -> str:
        completion = litellm_completion(
            model=self.model_name_or_path,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message['content']
    
    def generate(self, system_instruction, prompt, **kwargs) -> str:
        completion = litellm_completion(
            model=self.model_name_or_path,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return completion.choices[0].message['content']

class OpenAI_API(BaseLLM):
    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."

    def __init__(self, model_name_or_path) -> None:
        # make sure you set the OPENAI_API_KEY environment variable by ``export OPENAI_API_KEY=YOUR_API_KEY`` in command line
        # or you can set through ``os.environ['OPENAI_API_KEY'] = YOUR_API_KEY`` in the code
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError('OPENAI_API_KEY is not set')
        self.model_name_or_path = model_name_or_path
    
    def chat(self, messages, **kwargs) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model_name_or_path,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message['content']
    
    def generate(self, system_instruction, prompt, **kwargs) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model_name_or_path,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return completion.choices[0].message['content']

class ChatGLM2(BaseLLM):
    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."

    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
    
    def chat(self, messages, **kwargs):
        messages = [{"role": messages[1]["role"], "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
        history = [(query['content'], response['content']) for query, response in zip(messages[::2], messages[1::2])]
        response, history = self.model.chat(self.tokenizer, messages[-1]['content'], history=history, **kwargs)
        print(history)
        return response
    
    def generate(self, system_instruction, prompt, **kwargs):
        response, history = self.model.chat(self.tokenizer, system_instruction+"\n\n"+prompt, history=[], **kwargs)
        print(history)
        return response

class HFModels(BaseLLM):
    DEFAULT_GENERATION_KWARGS = {
        "do_sample": True,
        "max_new_tokens": 256,
    } # aligned with OpenAI API
    # See https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig for more parameters
    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."

    def __init__(self, model_name_or_path) -> None:
        # login through command line "huggingface-cli login" to assess some models such as LLaMa-2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        """Need transformers>=4.31.0 to use Llama-2."""
        self.pipeline = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def create_chat_prompt_and_eos_token(self, messages):
        prompt = messages[0]['content'] + "\n\n"
        for m in messages[1:]:
            prompt += f"{m['role']}: {m['content']}\n"
        prompt += f"assistant: "
        return prompt, '\n'

    def chat(self, messages, **kwargs):
        prompt, eos_token = self.create_chat_prompt_and_eos_token(messages)
        sequences = self.pipeline(
            prompt,
            prefix=None, # do not add additional special tokens
            eos_token_id=self.tokenizer.convert_tokens_to_ids(eos_token),
            **{**self.DEFAULT_GENERATION_KWARGS, **kwargs}
        )
        response = sequences[0]['generated_text'][len(prompt):].strip()
        return response
        
    def generate(self, system_instruction, prompt, **kwargs):
        prompt = (system_instruction+"\n\n"+prompt).strip()
        sequences = self.pipeline(
            prompt,
            eos_token_id=self.tokenizer.eos_token_id,
            **{**self.DEFAULT_GENERATION_KWARGS, **kwargs}
        )
        response = sequences[0]['generated_text'][len(prompt):].strip()
        return response

class LLaMa2(HFModels):
    DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def create_chat_prompt_and_eos_token(self, messages):
        """https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L213"""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>", "</s>"

        messages = [{"role": messages[1]["role"], "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]}] + messages[2:]

        messages_list = [
            f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
            for prompt, answer in zip(messages[::2], messages[1::2])
        ]
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

        return "".join(messages_list), self.tokenizer.eos_token
    

if __name__ == '__main__':
    # model = LLM('openai', 'gpt-3.5-turbo')
    # model = LLM('huggingface', 'Llama-2-7b-chat-hf')
    model = LLM('huggingface', 'chatglm2-6b')
    print(model.__dict__)
    print(model.chat("Help me to find a Chinese restaurant in Beijing."))
    print(model.chat("Great"))
    print(model.messages)
    model.set_system_instruction("Let's play a game. I'll start my sentence with 'fortunately' and you should go on but start the sentence with 'unfortunately'")
    print(model.generate('fortunately, you are an AI.'))
    