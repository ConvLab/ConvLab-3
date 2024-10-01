'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab.dialog_agent.agent import DialogueAgent
from convlab.e2e.emotod.emollama import EMOLLAMAAgent

class SimpleLLAMA(DialogueAgent):

    def __init__(self):

        nlu = None
        dst = None
        nlg = None
        simplellama_path = "/home/shutong/Emo-TOD/OUT_llama-2-7b-chat-hf/training_outputs/simple/4e-05_42_rank32/checkpoint-4000/"

        sys_policy = EMOLLAMAAgent(model_file=simplellama_path, simple=True)

        super().__init__(nlu, dst, sys_policy, nlg)

        self.agent_name = "Emo-LLAMA"

if __name__ == '__main__':

    agent = SimpleLLAMA()
    print("Loaded Agent successfully.")
