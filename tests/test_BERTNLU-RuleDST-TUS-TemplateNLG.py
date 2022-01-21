# available NLU models
# from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.policy.dqn.multiwoz.dqn_policy import DQNPolicy
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
# from convlab2.nlu.milu.multiwoz import MILU
# available DST models
from convlab2.dst.rule.multiwoz import RuleDST
# from convlab2.dst.mdbt.multiwoz import MDBT
# from convlab2.dst.sumbt.multiwoz import SUMBT
# from convlab2.dst.trade.multiwoz import TRADE
# from convlab2.dst.comer.multiwoz import COMER
# available Policy models
from convlab2.policy.rule.multiwoz import RulePolicy
# from convlab2.policy.ppo.multiwoz import PPOPolicy
# from convlab2.policy.pg.multiwoz import PGPolicy
# from convlab2.policy.mle.multiwoz import MLEPolicy
# from convlab2.policy.vhus.multiwoz import UserPolicyVHUS
# from convlab2.policy.mdrg.multiwoz import MDRGWordPolicy
# from convlab2.policy.hdsa.multiwoz import HDSA
# from convlab2.policy.larl.multiwoz import LaRL
# available NLG models
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.nlg.sclstm.multiwoz import SCLSTM
# available E2E models
# from convlab2.e2e.sequicity.multiwoz import Sequicity
# from convlab2.e2e.damd.multiwoz import Damd
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer
from pprint import pprint
import random
import json
import numpy as np
import torch

from convlab2.dst.rule.multiwoz.usr_dst import UserRuleDST
from convlab2.policy.tus.multiwoz.TUS import UserPolicy


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():
    # go to README.md of each model for more information
    # BERT nlu
    sys_nlu = BERTNLU()
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy = RulePolicy()
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    # assemble
    sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # specify the user config
    user_config = "/home/lubis/gpfs/project/lubis/convlab-2/pre-trained-models/TUS/exp/exp-24.json"
    user_mode = ""
    # BERT nlu trained on sys utterance
    user_nlu = BERTNLU()
    user_dst = UserRuleDST()
    # rule policy
    user_config = json.load(open(user_config))

    if user_mode:
        user_config["model_name"] = f"{user_config['model_name']}-{user_mode}"
    user_policy = UserPolicy(user_config)
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble
    user_agent = PipelineAgent(
        user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    analyzer.comprehensive_analyze(
        sys_agent=sys_agent, model_name='BERTNLU-RuleDST-TUS-TemplateNLG', total_dialog=1000)


if __name__ == '__main__':
    test_end2end()
