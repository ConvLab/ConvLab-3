'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab.dialog_agent.agent import DialogueAgent
from convlab.e2e.emotod.emollama import EMOLLAMAAgent

class EmoLLAMA(DialogueAgent):

    def __init__(self):

        nlu = None
        dst = None
        nlg = None
        emollama_path = "/home/shutong/Emo-TOD/OUT_llama-2-7b-chat-hf-1/training_outputs/emo_prev_conduct/4e-05_42_rank32/checkpoint-3500"

        sys_policy = EMOLLAMAAgent(model_file=emollama_path, simple=False)

        super().__init__(nlu, dst, sys_policy, nlg)

        self.agent_name = "Emo-LLAMA"

if __name__ == '__main__':

    agent = EmoLLAMA()
    print("Loaded Agent successfully.")
