from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
from convlab.policy.vtrace_rnn_action_embedding import VTRACE_RNN
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.dialog_agent import PipelineAgent
from convlab.util.analysis_tool.analyzer import Analyzer
import random
import os
import json
import numpy as np
import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end(seed=20200202, n_dialogues=1000, args=None):

    if args.use_uncertain_query:
        args.use_belief_probs = True

    # Dialogue System
    sys_nlu = None
    sys_dst = SetSUMBTTracker(model_type='roberta',
                        model_path=args.setsumbt_path,
                        get_belief_state_probs=args.use_belief_probs,
                        use_uncertain_query=args.use_uncertain_query)
    sys_policy = VTRACE_RNN(is_train=False, use_masking=False, shrink=False, noisy=True, action_embedding=True, full_noise=False)
    if args.use_uncertain_query:
        sys_policy.vector.setup_uncertain_query(sys_dst.thresholds)
    sys_policy.load(args.policy_path)
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # User Simulator
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                       model_file='/gpfs/project/niekerk/data/bert_multiwoz_sys_context.zip')
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True, label_noise=args.user_label_noise, text_noise=args.user_text_noise)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(seed)
    name = 'SetSUMBT'
    name = name + '-DistilledEnsemble' if 'distilled' in args.setsumbt_path else name + '-SingleModel'
    name = name + '-Belief' if args.use_belief_probs else name
    name = name + '-UncertainDBSearch' if args.use_uncertain_query else name
    name += '-VTRACE_RNN-TemplateNLG'
    name = name + '-UserLabelNoise' if args.user_label_noise else name
    name = name + '-UserTextNoise' if args.user_text_noise else name
    name += f'-Seed{seed}'
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=name, total_dialog=n_dialogues)


def get_args(args):
    path = args.experiment_path.split('/')[-1].split('_', 1)[-1] + '.json'
    reader = open(os.path.join(args.experiment_path, 'configs', path), 'r')
    exp_args = json.load(reader)['args']
    reader.close()

    args.seed = exp_args["seed"] if args.seed is None else args.seed
    args.setsumbt_path = exp_args["setsumbt_path"]
    args.use_belief_probs = exp_args["use_belief_probs"]
    args.use_uncertain_query = exp_args["use_uncertain_query"]
    args.user_label_noise = exp_args["user_label_noise"]
    args.user_text_noise = exp_args["user_text_noise"]
    args.policy_path = os.path.join(args.experiment_path, 'save', 'vtrace_rnn')
    return args


if __name__ == '__main__':
    # Get arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='Seed', default=None, type=int)
    parser.add_argument('--n_dialogues', help='Number of eval dialogues', default=1000, type=int)
    parser.add_argument('--experiment_path', help='RL experiment results directory', type=str)
    args = parser.parse_args()

    args = get_args(args)

    test_end2end(seed=args.seed, n_dialogues=args.n_dialogues, args=args)
