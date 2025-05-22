from argparse import ArgumentParser

from convlab.dialog_agent import PipelineAgent
from convlab.policy.emoUS_v2.langEmoUS import UserPolicy
from convlab.policy.vector.vector_nodes import VectorNodes
from convlab.policy.vtrace_DPT import VTRACE
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.dialog_agent.session import BiSession
from convlab.dialog_agent.env import Environment
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU
from convlab.policy.mle import MLE
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.policy.ppo import PPO


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str,
                        default="convlab/policy/emoUS/unify/default/EmoUS_default")
    parser.add_argument("--mode", type=str, default="language")
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


def test():
    args = arg_parser()
    use_sentiment, emotion_mid = False, False
    usr_policy = UserPolicy(
        model_checkpoint=args.model_checkpoint,
        mode=args.mode,
        sample=args.sample,
        use_sentiment=use_sentiment,
        emotion_mid=emotion_mid,
        model_type="encoder_decoder")
    usr_nlu = BERTNLU(mode="sys", config_file="multiwoz21_sys_context3.json",
                      model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_system_context3.zip")
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')

    sys_nlu = BERTNLU(mode="user", config_file="multiwoz21_all_context3.json",
                      model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_all_context3.zip")
    sys_dst = RuleDST()
    sys_policy = PPO(dataset_name="multiwoz21", load_path="from_pretrained")
    sys_nlg = TemplateNLG(is_user=False)
    sys = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    usr.init_session()
    sys.init_session()
    usr_utt = usr.response("what can I help you?")
    sys_utt = sys.response(usr_utt)
    print(usr_utt, sys_utt)
    usr_utt = usr.response(sys_utt)
    sys_utt = sys.response(usr_utt)
    print(usr_utt, sys_utt)


if __name__ == "__main__":
    test()
