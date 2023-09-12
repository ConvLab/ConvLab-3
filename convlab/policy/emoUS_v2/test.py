from argparse import ArgumentParser

from convlab.dialog_agent import PipelineAgent
from convlab.policy.emoUS_v2.semanticEmoUS import UserPolicy
from convlab.policy.vector.vector_nodes import VectorNodes
from convlab.policy.vtrace_DPT import VTRACE
from convlab.dst.rule.multiwoz.dst import RuleDST
from convlab.dialog_agent.session import BiSession
from convlab.dialog_agent.env import Environment


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
    usr = PipelineAgent(None, None, usr_policy, None, name='user')
    usr.init_session()
    print(usr.response([], sys_conduct="neutral"))
    print(usr.response([], sys_conduct="enthusiastic"))

    vectorizer = VectorNodes(dataset_name='multiwoz21',
                             use_masking=True,
                             manually_add_entity_names=True,
                             seed=0,
                             filter_state=True)
    sys_policy = VTRACE(is_train=False,
                        seed=0,
                        vectorizer=vectorizer,
                        load_path="from_pretrained")
    # test for seestion
    dst = RuleDST()
    sys = PipelineAgent(None, dst, sys_policy, None, name='sys')
    print("Create session")
    sess = BiSession(sys, usr)
    print("get next turn")
    x = sess.next_turn([])
    print(x)

    print("="*20)
    env = Environment(sys_nlg=None, usr=usr, sys_nlu=None, sys_dst=dst)
    s = env.reset()
    sys_act = sys_policy.predict(s)
    sys_conduct = sys_policy.get_conduct()
    s = env.step(action=sys_act, sys_conduct=sys_conduct)
    print(x)


if __name__ == "__main__":
    test()
