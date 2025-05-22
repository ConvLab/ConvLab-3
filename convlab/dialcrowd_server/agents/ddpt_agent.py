'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab.dialcrowd_server.agents.base_agent import BaseAgent
from convlab.policy.vtrace_DPT import VTRACE
from convlab.util.custom_util import get_config


class DDPTAgent(BaseAgent):

    def __init__(self):

        config_path = "convlab/policy/vtrace_DPT/configs/SetSUMBT-TemplateNLG.json"
        load_path = "convlab/policy/vtrace_DPT/supervised"
        conf = get_config(config_path, [])

        policy = VTRACE(vectorizer=conf['vectorizer_sys_activated'],
                        load_path=load_path,
                        is_train=False)

        super().__init__(conf, policy)

        self.agent_name = "DDPT"


if __name__ == '__main__':
    agent = DDPTAgent()
    print("Loaded Agent successfully.")
