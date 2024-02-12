from argparse import ArgumentParser
from convlab.util import load_dataset
from convlab.util.custom_util import set_seed, get_config, env_config, create_goals, data_goals
import torch
import json
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str,
                        default="DDPT", help="name of model")
    parser.add_argument("--config", "-c", type=str,
                        default='', help="model config")
    parser.add_argument("--weight", "-w", type=str,
                        default='', help="model weight")
    parser.add_argument("--pipeline", "-p", type=str,
                        default='', help="system pipeline")
    parser.add_argument("--output", "-o", type=str,
                        default='', help="output file name")

    return parser.parse_args()


def init(pipeline, model, config, weight=""):
    seed = 0
    set_seed(seed)

    conf = get_config(pipeline, [])

    if model == "DDPT":
        from convlab.policy.vtrace_DPT import VTRACE
        policy_sys = VTRACE(
            is_train=False, config_path=config, vectorizer=conf['vectorizer_sys_activated'])

    try:
        if weight:
            policy_sys.load(weight)
        else:
            policy_sys.load(conf['model']['load_path'])
    except Exception as e:
        print(f"Could not load a policy: {e}")

    env, sess = env_config(conf, policy_sys)
    return env, sess


def generate_dialog(system, dialog):
    system.init_session()
    dst = system.dst
    policy = system.policy
    nlg = system.nlg
    result = []
    for turn in dialog["turns"]:
        if turn["speaker"] == "user":
            user_utt = turn["utterance"]
            state = dst.update(user_utt)
            user_emotion = state['user_emotion']
            action = policy.predict(state)
            conduct = policy.get_conduct()
            sys_utt = nlg.generate(
                action, conduct, user_utt)
            dst.state['history'].append(["usr", user_utt])

        else:
            sys_ref = turn["utterance"]
            dst.state['history'].append(["sys", sys_ref])
            result.append({
                "utt_idx": f'{dialog["original_id"]}_{turn["utt_idx"]}',
                "state": state,
                "user_utt": user_utt,
                "user_emotion": user_emotion,
                "sys_act": action,
                "sys_conduct": conduct,
                "sys_utt": sys_utt,
                "sys_ref": sys_ref
            })

    return result


def evaluate(pipeline, model, config, weight="", output="result.json"):
    env, sess = init(pipeline, model, config, weight)
    system = sess.sys_agent

    data = load_dataset("multiwoz21")
    # for m in data:
    for m in ["test"]:
        result = []
        for d in data[m]:
            result += generate_dialog(system, d)
        folder = os.path.dirname(output)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        basename = os.path.basename(output)
        file_name = os.path.join(folder, f"{m}-{basename}")

        json.dump(result, open(file_name, "w"), indent=2)


if __name__ == "__main__":
    args = arg_parser()
    evaluate(args.pipeline, args.model, args.config, args.weight, args.output)
