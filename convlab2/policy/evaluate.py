# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.dst.rule.multiwoz.usr_dst import UserRuleDST
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.policy.tus.multiwoz.TUS import UserPolicy
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.util.custom_util import set_seed


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


def evaluate(args, model_name, load_path, verbose=False):
    seed = 0
    set_seed(seed)

    dst_sys = RuleDST()

    if model_name == "PPO":
        from convlab2.policy.ppo import PPO
        if load_path:
            policy_sys = PPO(False)
            policy_sys.load(load_path)
        else:
            policy_sys = PPO.from_pretrained()
    elif model_name == "RULE":
        policy_sys = RulePolicy()
    elif model_name == "PG":
        from convlab2.policy.pg import PG
        if load_path:
            policy_sys = PG(False)
            policy_sys.load(load_path)
        else:
            policy_sys = PG.from_pretrained()
    elif model_name == "MLE":
        from convlab2.policy.mle import MLE
        if load_path:
            policy_sys = MLE()
            policy_sys.load(load_path)
        else:
            policy_sys = MLE.from_pretrained()
    elif model_name == "GDPL":
        from convlab2.policy.gdpl import GDPL
        if load_path:
            policy_sys = GDPL(False)
            policy_sys.load(load_path)
        else:
            policy_sys = GDPL.from_pretrained()
    user_type = args.user.lower()
    if user_type == "rule":
        dst_usr = None
        policy_usr = RulePolicy(character='usr')
    elif user_type == "tus":
        dst_usr = UserRuleDST()
        user_config = json.load(open(args.user_config))
        policy_usr = UserPolicy(user_config)
    elif user_type == "vhus":
        from convlab2.policy.vhus.multiwoz import UserPolicyVHUS
        dst_usr = None
        policy_usr = UserPolicyVHUS(
            load_from_zip=True, model_file="/home/linh/convlab-2/vhus_simulator_multiwoz.zip")

    simulator = PipelineAgent(None, dst_usr, policy_usr, None, 'user')
    agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

    evaluator = MultiWozEvaluator()
    sess = BiSession(agent_sys, simulator, None, evaluator)

    action_dict = {}

    task_success = {'Complete': [], 'Success': [], 'Success strict': [], 'total_return': []}
    for seed in range(1000, 1400):
        set_seed(seed)
        sess.init_session()
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
            total_return += evaluator.get_reward(session_over)

            if session_over:
                task_succ = sess.evaluator.task_success()
                task_succ = sess.evaluator.success
                task_succ_strict = sess.evaluator.success_strict
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

    for key in task_success:
        logging.info(
            f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="PPO", help="name of model")
    parser.add_argument("--load_path", type=str,
                        default='', help="path of model")
    parser.add_argument("--log_path_suffix", type=str,
                        default="", help="suffix of path of log file")
    parser.add_argument("--log_dir_path", type=str,
                        default="log", help="path of log directory")
    parser.add_argument("--user_config", type=str,
                        default="convlab2/policy/tus/multiwoz/exp/default.json")
    parser.add_argument("--user_mode", type=str, default="")
    parser.add_argument("--user", type=str, default="rule")
    parser.add_argument("--verbose", action='store_true', help="whether to output utterances")

    args = parser.parse_args()

    init_logging(log_dir_path=args.log_dir_path,
                 path_suffix=args.log_path_suffix)
    evaluate(
        args=args,
        model_name=args.model_name,
        load_path=args.load_path,
        verbose=args.verbose
    )
