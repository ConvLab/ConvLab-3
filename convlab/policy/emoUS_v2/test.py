from argparse import ArgumentParser

from convlab.dialog_agent import PipelineAgent
from convlab.policy.emoUS_v2.langEmoUS import UserPolicy
from convlab.policy.vector.vector_nodes import VectorNodes
from convlab.policy.vtrace_DPT import VTRACE
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.dialog_agent.session import BiSession
from convlab.dialog_agent.env import Environment
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU


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
    nlu = BERTNLU(mode="sys", config_file="multiwoz21_sys_context3.json",
                  model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_system_context3.zip")
    usr = PipelineAgent(nlu, None, usr_policy, None, name='user')
    usr.init_session()
    print(usr.response("what can I help you?"))
    print(usr.response("the restaurant area is north"))


if __name__ == "__main__":
    test()
