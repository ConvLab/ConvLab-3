from argparse import ArgumentParser
import json

from convlab.dialog_agent import PipelineAgent
# from convlab.policy.llmforus.llmforus import UserPolicy
from convlab.policy.ppo import PPO
from convlab.task.multiwoz.goal_generator import GoalGenerator
from tqdm import tqdm
from convlab.util.custom_util import get_config, set_seed, env_config, create_goals


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--peft-checkpoint", type=str)
    parser.add_argument("--model-config", type=str,
                        default="convlab/policy/llmforus/configs/BertNLU-RuleDST-PPOpolicy-TemplateNLG.json")
    parser.add_argument("--policy-config", type=str,
                        default="convlab/policy/llmforus/configs/policy.json")
    parser.add_argument("--num-goals", type=int, default=2)
    return parser.parse_args()


def get_agents(model_config):
    conf = get_config(model_config, [])
    policy_sys = PPO(vectorizer=conf['vectorizer_sys_activated'])
    policy_sys.load(conf['model']['load_path'])
    sys = PipelineAgent(conf['nlu_sys_activated'],
                        conf['dst_sys_activated'],
                        policy_sys,
                        conf['sys_nlg_activated'],
                        name="sys",
                        debug=True)
    'nlu_usr', 'dst_usr', 'policy_usr', 'usr_nlg'
    usr = PipelineAgent(conf['nlu_usr_activated'],
                        conf['dst_usr_activated'],
                        conf['policy_usr_activated'],
                        conf['usr_nlg_activated'],
                        name="user",
                        debug=True)
    return sys, usr


def get_system_policy(model_config):
    conf = get_config(model_config, [])
    policy_sys = PPO(vectorizer=conf['vectorizer_sys_activated'])
    policy_sys.load(conf['model']['load_path'])
    return policy_sys


if __name__ == "__main__":

    set_seed(20220220)

    args = arg_parser()
    # sys, usr = get_agents(args.model_config)
    policy_sys = get_system_policy(args.policy_config)
    print("policy_config", args.policy_config)
    print(args.model_config)
    conf = get_config(args.model_config, [])
    env, sess = env_config(conf, policy_sys)
    goal_generator = GoalGenerator()
    num_goals = args.num_goals
    goals = create_goals(goal_generator, num_goals=num_goals,
                         single_domains=False, allowed_domains=None)
    conversation = []
    for seed in tqdm(range(1000, 1000 + num_goals)):
        dialogue = {"seed": seed, "log": []}
        set_seed(seed)
        sess.init_session(goal=goals[seed-1000])
        sys_response = ""
        sys_action = []
        actions = 0.0
        total_return = 0.0
        turns = 0
        task_succ = 0
        task_succ_strict = 0
        complete = 0
        dialogue["goal"] = env.usr.policy.policy.goal.domain_goals
        dialogue["user info"] = env.usr.policy.policy.user_info

        for i in range(40):
            sys_response, sys_action, user_response, session_over, reward = sess.next_turn_two_way(
                sys_response, sys_action)
            dialogue["log"].append(
                {"role": "usr",
                 "utt": user_response,
                 "emotion": env.usr.policy.policy.emotion,
                 "act": env.usr.policy.policy.semantic_action})
            dialogue["log"].append(
                {"role": "sys",
                 "utt": sys_response,
                 "act": sys_action})

            # logging.info(f"Actions in turn: {len(sys_response)}")
            turns += 1
            total_return += sess.evaluator.get_reward(session_over)

            if session_over:
                task_succ = sess.evaluator.task_success()
                task_succ = sess.evaluator.success
                task_succ_strict = sess.evaluator.success_strict
                complete = sess.evaluator.complete
                break

        dialogue['Complete'] = complete
        dialogue['Success'] = task_succ
        dialogue['Success strict'] = task_succ_strict
        dialogue['total_return'] = total_return
        dialogue['turns'] = turns

        conversation.append(dialogue)

    with open("convlab/policy/llmforus/conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)
    # print(sys.response("I want to find a hotel in the north"))

    # usr_policy = UserPolicy(model_checkpoint=args.model_checkpoint,
    #                         peft_checkpoint=args.peft_checkpoint)
    # usr = PipelineAgent(None, None, usr_policy, None, name='user')
    # print(usr.policy.get_goal())

    # print(usr.response("Hi, what can I help you?"))
    # print("emotion", usr.policy.get_emotion())
    # print("action", usr.policy.semantic_action)
