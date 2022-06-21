from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
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


def test_end2end(seed=20200202, n_dialogues=1000, label_noise=0.0, text_noise=0.0, dst_model_path=None, nlu_model_path=None):

    # Dialogue System
    sys_nlu = None
    sys_dst = SetSUMBTTracker(model_type='roberta', model_path=dst_model_path,
                            get_belief_state_probs=args.use_uncertain_query,
                            use_uncertain_query=args.use_uncertain_query)
    sys_policy = RulePolicy()
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # User Simulator
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', model_file=nlu_model_path)
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True, label_noise=label_noise, text_noise=text_noise, seed=seed)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(seed)
    name = 'SetSUMBT-RulePolicy-TemplateNLG'
    name = name + '-UncertainQuery' if args.use_uncertain_query else name
    name = name + '-LabelNoise' if label_noise > 0.0 else name
    name = name + '-TextNoise' if text_noise > 0.0 else name
    name += f'-Seed{seed}'
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=name, total_dialog=n_dialogues)

if __name__ == '__main__':
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Seed', default=20200202, type=int)
    parser.add_argument('--n_dialogues', help='Number of eval dialogues', default=1000, type=int)
    parser.add_argument('--user_label_noise', help='User Label Noise', default=0.0, type=float)
    parser.add_argument('--user_text_noise', help='User Text Noise', default=0.0, type=float)
    parser.add_argument("--use_uncertain_query", action='store_true')
    parser.add_argument('--dst_model_path', type=str)
    parser.add_argument('--nlu_model_path', type=str)
    args = parser.parse_args()

    test_end2end(seed=args.seed, n_dialogues=args.n_dialogues, label_noise=args.user_label_noise,
                text_noise=args.user_text_noise, dst_model_path=args.dst_model_path, nlu_model_path=args.nlu_model_path)
