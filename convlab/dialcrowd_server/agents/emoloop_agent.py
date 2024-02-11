'''

Build up an pipeline agent with nlu, dst, policy and nlg.

@author: Chris Geishauser
'''

from convlab.dialcrowd_server.agents.base_agent import BaseAgent
from convlab.policy.vtrace_DPT import VTRACE
from convlab.util.custom_util import get_config


class EmoLoopAgent(BaseAgent):

    def __init__(self):

        pipeline_config = "/home/shutong/ConvLab3/convlab/dialcrowd_server/agents/emoloop_pipeline.json"
        policy_config = "/home/shutong/ConvLab3/convlab/dialcrowd_server/agents/emoloop_ddpt_config.json"
        policy_model_path = "/home/shutong/models/acl2024_ckpts/emoloop_policy"
        conf = get_config(pipeline_config, [])

        policy = VTRACE(is_train=False, load_path=policy_model_path, config_path=policy_config, vectorizer=conf['vectorizer_sys_activated'])

        super().__init__(conf, policy)

        self.agent_name = "EmoLoop"


if __name__ == '__main__':
    agent = EmoLoopAgent()
    print("Loaded Agent successfully.")
