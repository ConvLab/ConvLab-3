'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab.dialog_agent.agent import DialogueAgent
from convlab.e2e.emotod.emotod import EMOTODAgent

class EmoTOD(DialogueAgent):

    def __init__(self):

        nlu = None
        dst = None
        nlg = None
        sys_policy = EMOTODAgent(model_file='/home/shutong/models/emotod')

        super().__init__(nlu, dst, sys_policy, nlg)

        self.agent_name = "Emo-TOD"

if __name__ == '__main__':

    agent = EmoTOD()
    print("Loaded Agent successfully.")
