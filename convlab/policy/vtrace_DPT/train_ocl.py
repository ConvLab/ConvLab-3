# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: Chris Geishauser
"""

import sys
import os
import logging
import time
import torch
import numpy as np
import json

from copy import deepcopy
from torch import multiprocessing as mp
from argparse import ArgumentParser
from convlab.policy.vtrace_DPT import VTRACE
from convlab.policy.vtrace_DPT.memory import Memory
from convlab.policy.vtrace_DPT.multiprocessing_helper import get_queues, start_processes, submit_jobs, \
    terminate_processes
from convlab.policy.vtrace_DPT.ocl.ocl_helper import load_budget, check_setup, log_used_budget, get_goals, \
    update_online_metrics
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import set_seed, init_logging, save_config, move_finished_training, env_config, \
    eval_policy, load_config_file, get_config
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def create_episodes(environment, policy, num_episodes, memory, goals):
    sampled_num = 0
    traj_len = 40
    metrics = []

    while sampled_num < num_episodes and goals:
        goal = goals.pop()
        s = environment.reset(goal)
        rl_return = 0

        user_act_list, sys_act_list, s_vec_list, action_list, reward_list, small_act_list, action_mask_list, mu_list, \
        trajectory_list, vector_mask_list, critic_value_list, description_idx_list, value_list, current_domain_mask, \
        non_current_domain_mask = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for t in range(traj_len):

            s_vec, mask = policy.vector.state_vectorize(s)
            with torch.no_grad():
                a = policy.predict(s)

            # s_vec_list.append(policy.info_dict['kg'])
            action_list.append(policy.info_dict['big_act'].detach())
            small_act_list.append(policy.info_dict['small_act'])
            action_mask_list.append(policy.info_dict['action_mask'])
            mu_list.append(policy.info_dict['a_prob'].detach())
            critic_value_list.append(policy.info_dict['critic_value'])
            vector_mask_list.append(torch.Tensor(mask))
            description_idx_list.append(policy.info_dict["description_idx_list"])
            value_list.append(policy.info_dict["value_list"])
            current_domain_mask.append(policy.info_dict["current_domain_mask"])
            non_current_domain_mask.append(policy.info_dict["non_current_domain_mask"])

            sys_act_list.append(policy.vector.action_vectorize(a))
            trajectory_list.extend([s['user_action'], a])

            # interact with env
            next_s, r, done = environment.step(a)
            rl_return += r
            reward_list.append(torch.Tensor([r]))

            next_s_vec, next_mask = policy.vector.state_vectorize(next_s)

            # update per step
            s = next_s

            if done:
                metrics.append({"success": environment.evaluator.success_strict, "return": rl_return,
                                  "avg_actions": torch.stack(action_list).sum(dim=-1).mean().item(),
                                  "turns": t, "goal": goal.domain_goals})
                memory.update_episode(description_idx_list, action_list, reward_list, small_act_list, mu_list,
                                      action_mask_list, critic_value_list, description_idx_list, value_list,
                                      current_domain_mask, non_current_domain_mask)
                break

        sampled_num += 1
    return metrics


def log_train_configs():
    logging.info('Train seed is ' + str(seed))
    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    logging.info(f"Number of processes for training: {train_processes}")


if __name__ == '__main__':

    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    begin_time = datetime.now()
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='convlab/policy/vtrace_DPT/ocl/semantic_level_config_ocl.json',
                        help="Load path for config file")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the policy parameter initialization")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")

    path = parser.parse_args().path
    seed = parser.parse_args().seed
    mode = parser.parse_args().mode
    save_eval = parser.parse_args().save_eval_dials

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)

    args = [('model', 'seed', seed)]

    environment_config = load_config_file(path)
    save_config(vars(parser.parse_args()), environment_config, config_save_path,
                json.load(open(os.path.dirname(__file__) + "/config.json", "r")))

    conf = get_config(path, args)
    seed = conf['model']['seed']
    set_seed(seed)

    policy_sys = VTRACE(is_train=True, seed=seed, vectorizer=conf['vectorizer_sys_activated'],
                        load_path=conf['model']['load_path'])
    policy_sys.share_memory()
    memory = Memory(seed=seed)
    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path

    env, sess = env_config(conf, policy_sys)

    # Setup uncertainty thresholding
    if env.sys_dst:
        try:
            if env.sys_dst.use_confidence_scores:
                policy_sys.vector.setup_uncertain_query(env.sys_dst.thresholds)
        except:
            logging.info('Uncertainty threshold not set.')

    # the timeline will decide how many dialogues are needed until a domain appears
    timeline, budget = load_budget(conf['model']['budget_path'])
    start_budget = deepcopy(budget)
    logging.info(f"Timeline: {timeline}")
    logging.info(f"Budget: {budget}")
    check_setup(timeline, budget)

    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    policy_sys.policy.action_embedder.forbidden_domains = []
    eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")
    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], 0)

    train_processes = conf['model']["process_num_train"]

    if train_processes > 1:
        # We use multiprocessing
        queues, episode_queues = get_queues(train_processes)
        online_metric_queue = mp.SimpleQueue()

    metric_keys = ["success", "return", "avg_actions", "turns", "goal"]
    online_metrics = {key: [] for key in metric_keys}
    num_dialogues = 0
    new_dialogues = conf['model']["new_dialogues"]
    training_done = False

    log_train_configs()
    goal_generator = GoalGenerator(domain_ordering_dist=dict((tuple(pair[0].split("-")), pair[1] / timeline['end'])
                                   for pair in start_budget))

    while not training_done:
        allowed_domains = [key for key, value in timeline.items() if value <= num_dialogues]
        forbidden_domains = [domain for domain in list(timeline.keys()) if domain not in allowed_domains]
        new_domain_introduced = len(allowed_domains) > len([key for key, value in timeline.items()
                                                            if value <= num_dialogues - new_dialogues])

        # we disable regularization for the first domain we see
        if len(allowed_domains) == 1:
            policy_sys.use_regularization = False
        else:
            policy_sys.use_regularization = True
        policy_sys.policy.action_embedder.forbidden_domains = forbidden_domains

        policy_sys.is_train = True
        if new_domain_introduced:
            # we sample a batch of goals until the next domain is introduced
            number_goals_required = min([value - num_dialogues for key, value in timeline.items()
                                         if value - num_dialogues > 0])

            logging.info(f"Creating {number_goals_required} goals..")
            goals, budget = get_goals(goal_generator, allowed_domains, budget, number_goals_required)
            logging.info("Goals created.")
            if train_processes > 1:
                # this means a new domain has appeared and the policy in the processes should be updated
                if len(allowed_domains) > 1:
                    # a new domain is introduced, first kill old processes before starting new
                    terminate_processes(processes, queues)

                processes = start_processes(train_processes, queues, episode_queues, env, policy_sys, seed,
                                            online_metric_queue)

        new_dialogues = conf['model']["new_dialogues"] if len(goals) > conf['model']["new_dialogues"] - 1 else len(goals)
        if train_processes == 1:
            metrics = create_episodes(env, policy_sys, new_dialogues, memory, goals)
        else:
            # create dialogues using the spawned processes
            time_now, metrics = submit_jobs(new_dialogues, queues, episode_queues, train_processes, memory, goals,
                                            online_metric_queue)
        num_dialogues += new_dialogues
        update_online_metrics(online_metrics, metrics, log_save_path, tb_writer)

        for r in range(conf['model']['update_rounds']):
            if num_dialogues > 50:
                policy_sys.update(memory)
                torch.cuda.empty_cache()

        if num_dialogues % 1000 == 0:
            logging.info(f"Online Metric" + '-' * 15 + f'Dialogues done: {num_dialogues}' + 15 * '-')
            for key in online_metrics:
                if key == "goal":
                    continue
                logging.info(f"{key}: {np.mean(online_metrics[key])}")
            log_used_budget(start_budget, budget)

        if num_dialogues % conf['model']['eval_frequency'] == 0:

            # run evaluation
            logging.info(f"Evaluating - " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-' * 60)
            policy_sys.is_train = False
            policy_sys.policy.action_embedder.forbidden_domains = []
            eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
            for key in eval_dict:
                tb_writer.add_scalar(key, eval_dict[key], num_dialogues)
            policy_sys.policy.action_embedder.forbidden_domains = forbidden_domains

        # if budget is empty and goals are used, training stops
        if sum([pair[1] for pair in budget]) == 0 and not goals:
            training_done = True

    policy_sys.save(save_path, f"end")

    logging.info("End of Training: " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    move_finished_training(dir_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))
