# available NLU models
# from convlab.nlu.svm.multiwoz import SVMNLU
# from convlab.policy.tus.multiwoz.TUS import UserPolicy
from convlab.dst.rule.multiwoz.usr_dst import UserRuleDST
import torch
import numpy as np
import json
import random
from pprint import pprint
from argparse import ArgumentParser
from convlab.nlu.jointBERT.unified_datasets import BERTNLU
# from convlab.nlu.jointBERT.multiwoz import BERTNLU as BERTNLU_woz
# from convlab.nlu.milu.multiwoz import MILU
# available DST models
from convlab.dst.rule.multiwoz import RuleDST
# from convlab.dst.mdbt.multiwoz import MDBT
# from convlab.dst.sumbt.multiwoz import SUMBT
# from convlab.dst.setsumbt.multiwoz.Tracker import SetSUMBTTracker
# from convlab.dst.trippy.multiwoz import TRIPPY
# from convlab.dst.trade.multiwoz import TRADE
# from convlab.dst.comer.multiwoz import COMER
# available Policy models
from convlab.policy.rule.multiwoz import RulePolicy
# from convlab.policy.ppo.multiwoz import PPOPolicy
# from convlab.policy.pg.multiwoz import PGPolicy。
# from convlab.policy.mle.multiwoz import MLEPolicy
# from convlab.policy.gdpl.multiwoz import GDPLPolicy
# from convlab.policy.vhus.multiwoz import UserPolicyVHUS
# from convlab.policy.mdrg.multiwoz import MDRGWordPolicy
# from convlab.policy.hdsa.multiwoz import HDSA
# from convlab.policy.larl.multiwoz import LaRL
from convlab.policy.lava.multiwoz import LAVA
# available NLG models
from convlab.nlg.template.multiwoz import TemplateNLG
# from convlab.nlg.sclstm.multiwoz import SCLSTM
# available E2E models
# from convlab.e2e.sequicity.multiwoz import Sequicity
# from convlab.e2e.damd.multiwoz import Damd
from convlab.dialog_agent import PipelineAgent, BiSession
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
import pdb  # ;pdb.set_trace()
from convlab.util.analysis_tool.analyzer import Analyzer
from argparse import ArgumentParser
from pprint import pprint
import random
import json
import numpy as np
import os
import torch
# Lin's US
# from convlab.dst.rule.multiwoz.usr_dst import UserRuleDST
# from convlab.policy.tus.multiwoz.NUS import UserPolicy


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end(args, model_dir):
    # BERT nlu
    if args.dst_type=="bertnlu_rule":
        sys_nlu = BERTNLU("user", config_file="multiwoz21_user_context3.json", model_file="bertnlu_unified_multiwoz21_user_context3")
    elif args.dst_type in ["setsumbt", "trippy"]:
        sys_nlu = None
    
    # simple rule DST
    # sys_dst = SUMBT()
    if args.dst_type=="bertnlu_rule":
        sys_dst = RuleDST()
    elif args.dst_type=="trippy":
        sys_dst = TRIPPY(model_type='roberta',
                    model_path='/gpfs/project/lubis/convlab-2/pre-trained-models/trippy-checkpoint-10647',
                    nlu_path='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_all_context.zip')
    elif args.dst_type=="setsumbt":
        sys_dst = SetSUMBTTracker(model_type='roberta',
                        model_path='/gpfs/project/lubis/convlab-2/pre-trained-models/end2')



    # where the models are saved from training
    # lava_dir = "/gpfs/project/lubis/ConvLab-3/convlab/policy/lava/multiwoz/experiments_woz/sys_config_log_model/"
    lava_dir = "/gpfs/project/lubis/LAVA_code/LAVA_published/experiments_woz/sys_config_log_model/"

    if "rl" in model_dir:
        lava_path = "{}/{}/reward_best.model".format(lava_dir, model_dir)
    else:
        # default saved model format
        model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(os.path.join(lava_dir, model_dir)) if 'model' in p and 'rl' not in p])
        best_epoch = model_ids[-1]
        lava_path = "{}/{}/{}-model".format(lava_dir, model_dir, best_epoch)

    print(lava_path)

    sys_policy = LAVA(lava_path)

    # template NLG
    sys_nlg = None

    # BERT nlu trained on sys utterance
    user_nlu = BERTNLU("sys", config_file="multiwoz21_system_context3_new.json", model_file="bertnlu_unified_multiwoz21_system_context3")
    if args.US_type == "ABUS":
        # not use dst
        user_dst = None
        # rule policy
        user_policy = RulePolicy(character='usr')
    elif args.US_type == "TUS":
        # specify the user config
        user_config = "/home/lubis/gpfs/project/lubis/convlab-2/pre-trained-models/TUS/exp/exp-24.json"
        user_mode = ""
        user_dst = UserRuleDST()
        # rule policy
        user_config = json.load(open(user_config))

        if user_mode:
            user_config["model_name"] = f"{user_config['model_name']}-{user_mode}"
        user_policy = UserPolicy(user_config)
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble agents
    user_agent = PipelineAgent(
        user_nlu, user_dst, user_policy, user_nlg, name='user')
    sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    sys_agent.add_booking_info = False


    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    #seed = 2020
    set_seed(args.seed)

    model_name = '{}_{}_lava_{}'.format(args.US_type, args.dst_type, model_dir)
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=model_name, total_dialog=500)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lava_dir", type=str, default="2020-05-12-14-51-49-actz_cat/rl-2020-05-18-10-50-48")
    parser.add_argument("--US_trained", type=bool, default=False, help="whether to use model trained on US or not")
    parser.add_argument("--seed", type=int, default=20200202, help="seed for random processes")
    parser.add_argument("--US_type", type=str, default="ABUS", help="which user simulator to us, ABUS or TUS")
    parser.add_argument("--dst_type", type=str, default="bertnlu_rule", help="which DST to use, bertnlu_rule, setsumbt, or trippy")
    args = parser.parse_args()

    test_end2end(args, args.lava_dir)

