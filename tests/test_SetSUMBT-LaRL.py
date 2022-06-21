from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.policy.larl.multiwoz import LaRL
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
    sys_nlu = None
    sys_dst = SetSUMBTTracker(model_type='roberta',
                        model_path='/gpfs/project/niekerk/results/nbt/convlab_setsumbt',
                        nlu_path='/gpfs/project/niekerk/data/bert_multiwoz_all_context.zip')
    sys_policy = LaRL(model_file='/gpfs/project/niekerk/data/larl.zip')
    sys_nlg = None
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
    name=f'SetSUMBT-LaRL-Seed{seed}'
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=name, total_dialog=n_dialogues)

if __name__ == '__main__':
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Seed', default=20211202, type=int)
    parser.add_argument('--n_dialogues', help='Number of eval dialogues', default=1000, type=int)
    args = parser.parse_args()

    test_end2end(seed=args.seed, n_dialogues=args.n_dialogues)
