from argparse import ArgumentParser

from convlab.dialog_agent import PipelineAgent
from convlab.policy.emoUS_v2.semanticEmoUS import UserPolicy


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
    print(usr.policy.policy.predict(sys_act=[], sys_conduct="neutral"))
    print(usr.policy.policy.predict(sys_act=[], sys_conduct="enthusiastic"))


if __name__ == "__main__":
    test()
