'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab2.dialcrowd_server.agents.base_agent import BaseAgent
from convlab2.policy.vtrace_DPT import VTRACE
from convlab2.util.custom_util import get_config


class DDPTAgent(BaseAgent):

    def __init__(self):

        config_path = ""
        conf = get_config(config_path, [])

        policy = VTRACE(vectorizer=conf['vectorizer_sys_activated'])
        policy.load(conf['model']['load_path'])

        super().__init__(conf, policy)

        self.agent_name = "DDPT"
