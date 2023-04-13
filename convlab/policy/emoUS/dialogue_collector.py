from argparse import ArgumentParser

from tqdm import tqdm

from convlab.policy.rule.multiwoz import RulePolicy
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import (create_goals, data_goals, env_config,
                                      get_config, set_seed)


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="the model path")
    parser.add_argument("-N", "--num", type=int,
                        default=500, help="# of evaluation dialogue")
    parser.add_argument("--model", type=str,
                        default="ppo", help="# of evaluation dialogue")
    return parser.parse_args()


def interact(model_name, config, seed=0, num_goals=500):
    conversation = []
    set_seed(seed)
    conf = get_config(config, [])

    if model_name == "rule":
        policy_sys = RulePolicy()
    elif model_name == "ppo":
        from convlab.policy.ppo import PPO
        policy_sys = PPO(vectorizer=conf['vectorizer_sys_activated'])

    model_path = conf['model']['load_path']
    if model_path:
        policy_sys.load(model_path)

    env, sess = env_config(conf, policy_sys)
    goal_generator = GoalGenerator()

    goals = create_goals(goal_generator, num_goals=num_goals,
                         single_domains=False, allowed_domains=None)

    for seed in tqdm(range(1000, 1000 + num_goals)):
        dialogue = {"seed": seed, "log": []}
        set_seed(seed)
        sess.init_session(goal=goals[seed-1000])
        sys_response = []
        actions = 0.0
        total_return = 0.0
        turns = 0
        task_succ = 0
        task_succ_strict = 0
        complete = 0
        dialogue["goal"] = env.usr.policy.policy.goal.domain_goals
        dialogue["user info"] = env.usr.policy.policy.user_info

        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response)
            dialogue["log"].append(
                {"role": "usr",
                 "utt": user_response,
                 "emotion": env.usr.policy.policy.emotion,
                 "act": env.usr.policy.policy.semantic_action})
            dialogue["log"].append({"role": "sys", "utt": sys_response})

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
    return conversation


if __name__ == "__main__":
    import json
    from datetime import datetime
    import os
    time = f"{datetime.now().strftime('%y-%m-%d-%H-%M')}"
    args = arg_parser()
    conversation = interact(model_name=args.model,
                            config=args.config,
                            num_goals=args.num)
    data = {"config": json.load(open(args.config)),
            "conversation": conversation}
    folder_name = os.path.join("convlab/policy/emoUS", "conversation")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    json.dump(data,
              open(os.path.join(folder_name, f"{time}.json"), 'w'),
              indent=2)
