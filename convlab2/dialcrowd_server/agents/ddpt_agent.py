'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab2.dialog_agent.agent import DialogueAgent
from convlab2.dst.setsumbt.multiwoz import SetSUMBTTracker
from convlab2.policy.vector.vector_nodes import VectorNodes
from convlab2.policy.vtrace_DPT import VTRACE
from convlab2.nlg.scgpt.multiwoz.scgpt import SCGPT


class Agent(DialogueAgent):

    def __init__(self):

        nlu = None
        dst = SetSUMBTTracker(model_path="end")

        vectorizer = VectorNodes(use_masking=True, manually_add_entity_names=True)
        policy_path = ""
        policy = VTRACE(load_path=policy_path, vectorizer=vectorizer)

        nlg_path = "scgpt"
        nlg = SCGPT(is_user=False, model_file=nlg_path)
        super().__init__(nlu, dst, policy, nlg)

        self.agent_name = "DDPT"
