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
import json

from torch import multiprocessing as mp
from argparse import ArgumentParser
from convlab.policy.vtrace_DPT import VTRACE
from convlab.policy.vtrace_DPT.memory import Memory
from convlab.policy.vtrace_DPT.multiprocessing_helper import get_queues, start_processes, submit_jobs, \
    terminate_processes
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import set_seed, init_logging, save_config, move_finished_training, env_config, \
    eval_policy, log_start_args, save_best, load_config_file, create_goals, get_config
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

emotion_dict = {"satisfied": 1, "neutral": 0,
                "dissatisfied": -1, "abusive": -1}


def create_episodes(environment, policy, num_episodes, memory, goals):
    sampled_num = 0
    traj_len = 40

    while sampled_num < num_episodes:
        goal = goals.pop()
        s = environment.reset(goal)
        prev_emotion = "none"

        user_act_list, sys_act_list, s_vec_list, action_list, reward_list, small_act_list, action_mask_list, mu_list, \
            trajectory_list, vector_mask_list, critic_value_list, description_idx_list, value_list, current_domain_mask, \
            non_current_domain_mask, use_temperature_list = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        emotion_temperature_list = []

        for t in range(traj_len):

            if hasattr(environment.sys_dst, 'get_emotion'):
                emotion = environment.sys_dst.get_emotion().lower()
                s['user_emotion'] = emotion
            elif hasattr(environment.usr.policy, 'get_emotion'):
                emotion = environment.usr.policy.get_emotion().lower()
                s['user_emotion'] = emotion
            else:
                emotion = "none"

            s_vec, mask = policy.vector.state_vectorize(s)
            with torch.no_grad():
                a = policy.predict(s)
                sys_conduct = policy.get_conduct()

            emotion_temperature_list.append(
                [emotion, policy.info_dict["temperature"], sys_conduct])

            # s_vec_list.append(policy.info_dict['kg'])
            action_list.append(policy.info_dict['big_act'].detach().cpu())
            small_act_list.append(policy.info_dict['small_act'].cpu())
            action_mask_list.append(policy.info_dict['action_mask'].cpu())
            mu_list.append(policy.info_dict['a_prob'].detach().cpu())
            critic_value_list.append(policy.info_dict['critic_value'].cpu())
            vector_mask_list.append(torch.Tensor(mask).cpu())
            description_idx_list.append(
                policy.info_dict["description_idx_list"].cpu())
            value_list.append(policy.info_dict["value_list"].cpu())
            current_domain_mask.append(
                policy.info_dict["current_domain_mask"].cpu())
            non_current_domain_mask.append(
                policy.info_dict["non_current_domain_mask"].cpu())
            use_temperature_list.append(torch.Tensor(
                [policy.info_dict["use_temperature"]]).cpu())

            sys_act_list.append(policy.vector.action_vectorize(a))
            trajectory_list.extend([s['user_action'], a])

            # interact with env
            next_s, r, done = environment.step(a, sys_conduct=sys_conduct)

            if policy.use_emotion_reward:
                emotion_reward = emotion_dict.get(emotion, 0)
                r += emotion_reward * policy.emotion_reward_weight - policy.emotion_reward_weight

            if policy.use_emotion_reward_difference:
                if hasattr(environment.usr.policy, 'get_emotion'):
                    emotion = environment.usr.policy.get_emotion().lower()
                    if prev_emotion != "none":
                        emotion_reward = emotion_dict.get(
                            emotion, 0) - emotion_dict.get(prev_emotion, 0)
                        # exclude the case where we go from satisfaction to neutral because that happens
                        if not (emotion_dict.get(emotion, 0) == 0 and emotion_dict.get(prev_emotion, 0) == 1):
                            r += emotion_reward
                    prev_emotion = emotion

            reward_list.append(torch.Tensor([r]))

            next_s_vec, next_mask = policy.vector.state_vectorize(next_s)

            # update per step
            s = next_s

            if done:
                memory.update_episode(description_idx_list, action_list, reward_list, small_act_list, mu_list,
                                      action_mask_list, critic_value_list, description_idx_list, value_list,
                                      current_domain_mask, non_current_domain_mask, use_temperature_list)
                break

        emotion_temperature_logs.append(emotion_temperature_list)

        sampled_num += 1


def log_train_configs():
    logging.info('Train seed is ' + str(seed))
    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    logging.info(f"Number of processes for training: {train_processes}")
    logging.info(f"Number of new dialogues per update: {new_dialogues}")
    logging.info(f"Number of total dialogues: {total_dialogues}")


if __name__ == '__main__':

    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    begin_time = datetime.now()
    parser = ArgumentParser()
    parser.add_argument("--config_name", type=str, default='RuleUser-Semantic-RuleDST',
                        help="Name of the configuration")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the policy parameter initialization")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--hyperparameter", type=str,
                        default="multiwoz21_dpt.json",)
    # We can specifiy the config file path or the config name
    if os.path.exists(parser.parse_args().config_name):
        path = parser.parse_args().config_name
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs',
                            f"{parser.parse_args().config_name}.json")
    seed = parser.parse_args().seed
    mode = parser.parse_args().mode
    save_eval = parser.parse_args().save_eval_dials

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if parser.parse_args().exp_dir is not None:
        exp_dir = os.path.join(exp_dir, parser.parse_args().exp_dir)

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(exp_dir, mode)

    args = [('model', 'seed', seed)] if seed is not None else list()

    environment_config = load_config_file(path)

    conf = get_config(path, args)
    seed = conf['model']['seed']
    set_seed(seed)

    policy_sys = VTRACE(is_train=True,
                        seed=seed,
                        vectorizer=conf['vectorizer_sys_activated'],
                        load_path=conf['model']['load_path'],
                        config_path=parser.parse_args().hyperparameter)
    policy_sys.share_memory()
    memory = Memory(seed=seed)
    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path

    save_config(vars(parser.parse_args()), environment_config,
                config_save_path, policy_config=policy_sys.cfg)

    env, sess = env_config(conf, policy_sys, check_book_constraints=conf['model'].get('check_book_constraints', True),
                           action_length_penalty=conf['model'].get('action_length_penalty', 0.0))

    if policy_sys.use_emotion_prediction:
        policy_sys.emotion_model = sess.user_agent.policy
        logging.info("Set emotion model for policy")

    # Setup uncertainty thresholding
    if env.sys_dst:
        try:
            if env.sys_dst.use_confidence_scores:
                policy_sys.vector.setup_uncertain_query(env.sys_dst.thresholds)
        except:
            logging.info('Uncertainty threshold not set.')

    single_domains = conf['goals']['single_domains']
    allowed_domains = conf['goals']['allowed_domains']
    logging.info(f"Single domains only: {single_domains}")
    logging.info(f"Allowed domains {allowed_domains}")
    logging.info(
        f"We check booking constraints: {conf['model'].get('check_book_constraints', True)}")
    logging.info(
        f"Action length penalty: {conf['model'].get('action_length_penalty', 0.0)}")

    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path,
                            single_domain_goals=single_domains, allowed_domains=allowed_domains)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")

    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], 0)
    best_complete_rate = eval_dict['complete_rate']
    best_success_rate = eval_dict['success_rate_strict']
    best_return = eval_dict['avg_return']

    train_processes = conf['model']["process_num_train"]

    if train_processes > 1:
        # We use multiprocessing
        queues, episode_queues = get_queues(train_processes)
        online_metric_queue = mp.SimpleQueue()
        processes = start_processes(train_processes, queues, episode_queues, env, policy_sys, seed,
                                    online_metric_queue)
    goal_generator = GoalGenerator()

    num_dialogues = 0
    new_dialogues = conf['model']["new_dialogues"]
    total_dialogues = conf['model']["total_dialogues"]

    emotion_temperature_logs = []

    log_train_configs()

    while num_dialogues < total_dialogues:

        goals = create_goals(goal_generator, new_dialogues, single_domains=single_domains,
                             allowed_domains=allowed_domains)
        if train_processes > 1:
            time_now, metrics = submit_jobs(new_dialogues, queues, episode_queues, train_processes, memory, goals,
                                            online_metric_queue)
        else:
            create_episodes(env, policy_sys, new_dialogues, memory, goals)
        num_dialogues += new_dialogues

        with open(os.path.join(save_path, 'emotion_temperature_logs.json'), 'w') as f:
            json.dump(emotion_temperature_logs, f)

        for r in range(conf['model']['update_rounds']):
            if num_dialogues > 50:
                # print("Updating policy")
                torch.cuda.empty_cache()
                policy_sys.update(memory)
                torch.cuda.empty_cache()

        if num_dialogues % conf['model']['eval_frequency'] == 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(
                f"Evaluating after Dialogues: {num_dialogues} - {time_now}" + '-' * 60)

            eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path,
                                    single_domain_goals=single_domains, allowed_domains=allowed_domains)

            best_complete_rate, best_success_rate, best_return = \
                save_best(policy_sys, best_complete_rate, best_success_rate, best_return,
                          eval_dict["complete_rate"], eval_dict["success_rate_strict"],
                          eval_dict["avg_return"], save_path)
            policy_sys.save(save_path, "last")
            for key in eval_dict:
                tb_writer.add_scalar(key, eval_dict[key], num_dialogues)

    logging.info("End of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    if train_processes > 1:
        terminate_processes(processes, queues)

    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        exp_dir, "finished_experiments"))
