from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.dst.rule.multiwoz import RuleDST
from convlab.policy.vtrace_rnn_action_embedding import VTRACE_RNN
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.dialog_agent import PipelineAgent
from convlab.util.analysis_tool.analyzer import Analyzer
import random
import numpy as np
import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end(seed=20200202, n_dialogues=1000):

    # Dialogue System
    # BERT nlu
    sys_nlu = BERTNLU()
    # simple rule DST
    sys_dst = RuleDST()
    sys_policy = VTRACE_RNN(is_train=False, shrink=True, noisy=True)
    sys_policy.load('/gpfs/project/niekerk/results/Convlab/vtrace_rnn_action_embedding_results/save/VTRACE_RNN_70000_0.99_noisy_shrink_2021-01-12-14-27-06')
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # User Simulator
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                       model_file='/gpfs/project/niekerk/data/bert_multiwoz_sys_context.zip')
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(seed)
    name=f'BERTNLU-RuleDST-VTRACE_RNN-TemplateNLG-Seed{seed}'
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=name, total_dialog=n_dialogues)

if __name__ == '__main__':
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Seed', default=20200202, type=int)
    parser.add_argument('--n_dialogues', help='Number of eval dialogues', default=1000, type=int)
    args = parser.parse_args()

    test_end2end(seed=args.seed, n_dialogues=args.n_dialogues)
