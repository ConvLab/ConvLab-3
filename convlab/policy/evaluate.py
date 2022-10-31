# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import os

import numpy as np
import torch
from convlab.dialog_agent.agent import PipelineAgent
from convlab.dialog_agent.session import BiSession
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import set_seed, get_config, env_config, create_goals, data_goals
from tqdm import tqdm


def init_logging(log_dir_path, path_suffix=None):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if path_suffix:
        log_file_path = os.path.join(
            log_dir_path, f"{current_time}_{path_suffix}.log")
    else:
        log_file_path = os.path.join(
            log_dir_path, "{}.log".format(current_time))
    print("LOG DIR PATH:", log_dir_path)
    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    format_str = "%(message)s"
    logging.basicConfig(level=logging.DEBUG, handlers=[
                        stderr_handler, file_handler], format=format_str)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(config_path, model_name, verbose=False, model_path="", goals_from_data=False, dialogues=500):
    seed = 0
    set_seed(seed)

    conf = get_config(config_path, [])

    if model_name == "PPO":
        from convlab.policy.ppo import PPO
        policy_sys = PPO(vectorizer=conf['vectorizer_sys_activated'])
    elif model_name == "RULE":
        policy_sys = RulePolicy()
    elif model_name == "PG":
        from convlab.policy.pg import PG
        policy_sys = PG(vectorizer=conf['vectorizer_sys_activated'])
    elif model_name == "MLE":
        from convlab.policy.mle import MLE
        policy_sys = MLE()
    elif model_name == "GDPL":
        from convlab.policy.gdpl import GDPL
        policy_sys = GDPL(vectorizer=conf['vectorizer_sys_activated'])
    elif model_name == "DDPT":
        from convlab.policy.vtrace_DPT import VTRACE
        policy_sys = VTRACE(is_train=False, vectorizer=conf['vectorizer_sys_activated'])

    try:
        if model_path:
            policy_sys.load(model_path)
        else:
            policy_sys.load(conf['model']['load_path'])
    except Exception as e:
        logging.info(f"Could not load a policy: {e}")

    env, sess = env_config(conf, policy_sys)
    action_dict = {}

    task_success = {'Complete': [], 'Success': [],
                    'Success strict': [], 'total_return': [], 'turns': []}

    goal_generator = GoalGenerator()
    if goals_from_data:
        logging.info("read goals from dataset...")
        goals = data_goals(dialogues, dataset="multiwoz21", dial_ids_order=0)
    else:
        logging.info("create goals from goal_generator...")
        goals = create_goals(goal_generator, num_goals=dialogues,
                             single_domains=False, allowed_domains=None)

    for seed in tqdm(range(1000, 1000 + dialogues)):
        set_seed(seed)
        sess.init_session(goal=goals[seed-1000])
        sys_response = []
        actions = 0.0
        total_return = 0.0
        turns = 0
        task_succ = 0
        task_succ_strict = 0
        complete = 0

        if verbose:
            logging.info("NEW EPISODE!!!!" + "-" * 80)
            logging.info(f"\n Seed: {seed}")
            logging.info(f"GOAL: {sess.evaluator.goal}")
            logging.info("\n")
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response)

            if verbose:
                logging.info(f"USER RESPONSE: {user_response}")
                logging.info(f"SYS RESPONSE: {sys_response}")

            actions += len(sys_response)
            length = len(sys_response)
            if length in action_dict:
                if sys_response not in action_dict[length]:
                    action_dict[length].append(sys_response)
            else:
                action_dict[length] = []
                action_dict[length].append(sys_response)

            # logging.info(f"Actions in turn: {len(sys_response)}")
            turns += 1
            total_return += sess.evaluator.get_reward(session_over)

            if session_over:
                task_succ = sess.evaluator.task_success()
                task_succ = sess.evaluator.success
                task_succ_strict = sess.evaluator.success_strict
                if goals_from_data:
                    complete = sess.user_agent.policy.policy.goal.task_complete()
                else:
                    complete = sess.evaluator.complete
                break

        if verbose:
            logging.info(f"Complete: {complete}")
            logging.info(f"Success: {task_succ}")
            logging.info(f"Success strict: {task_succ_strict}")
            logging.info(f"Return: {total_return}")
            logging.info(f"Average actions: {actions / turns}")

        task_success['Complete'].append(complete)
        task_success['Success'].append(task_succ)
        task_success['Success strict'].append(task_succ_strict)
        task_success['total_return'].append(total_return)
        task_success['turns'].append(turns)

    for key in task_success:
        logging.info(
            f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')
    logging.info(f"Average actions: {actions / turns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="PPO", help="name of model")
    parser.add_argument("-C", "--config_path", type=str,
                        default='', help="config path defining the environment for simulation and system pipeline")
    parser.add_argument("--model_path", type=str,
                        default='', help="if this is set, tries to load the model weights from this path"
                                         ", otherwise from config")
    parser.add_argument("-N", "--num_dialogues", type=int,
                        default=500, help="# of evaluation dialogue")
    parser.add_argument("-V", "--verbose", action='store_true',
                        help="whether to output utterances")
    parser.add_argument("--log_path_suffix", type=str,
                        default="", help="suffix of path of log file")
    parser.add_argument("--log_dir_path", type=str,
                        default="log", help="path of log directory")
    parser.add_argument("-D", "--goals_from_data", action='store_true',
                        help="load goal from the dataset")

    args = parser.parse_args()

    init_logging(log_dir_path=args.log_dir_path,
                 path_suffix=args.log_path_suffix)
    evaluate(config_path=args.config_path,
             model_name=args.model_name,
             verbose=args.verbose,
             model_path=args.model_path,
             goals_from_data=args.goals_from_data,
             dialogues=args.num_dialogues)
