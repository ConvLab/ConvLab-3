# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""

import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from torch import multiprocessing as mp

from convlab.policy.ppo import PPO
from convlab.policy.rlmodule import Memory
from convlab.util.custom_util import (env_config, eval_policy, get_config,
                                      init_logging, load_config_file,
                                      log_start_args, move_finished_training,
                                      save_best, save_config, set_seed)
from convlab.dialog_agent.env import Environment

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def sampler(pid, queue, evt, env, policy, num_dialogues, train_seed=0, user_reward=False):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()
    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    set_seed(train_seed)

    while sampled_traj_num < num_dialogues:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec, action_mask = policy.vector.state_vectorize(s)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)

            a = policy.predict(s)
            # print("---> sample action")
            # print(f"s     : {s['system_action']}")
            # print(f"a     : {a}")
            # interact with env
            next_s, r, done = env.step(a, user_reward=user_reward)
            # print(f"next_s: {next_s['system_action']}")

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec, next_action_mask = policy.vector.state_vectorize(
                next_s)
            next_s_vec = torch.Tensor(next_s_vec)

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(
                a), r, next_s_vec.numpy(), mask, action_mask.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, num_train_dialogues, process_num, seed, user_reward=False):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
        :param env:
        :param policy:
    :param batchsz:
        :param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_num_dialogues = np.ceil(num_train_dialogues / process_num).astype(np.int32)
    train_seeds = random.sample(range(0, 1000), process_num)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_num_dialogues, train_seeds[i], user_reward)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def update(env, policy, num_dialogues, epoch, process_num, seed=0, user_reward=False):

    # sample data asynchronously
    batch = sample(env, policy, num_dialogues, process_num, seed, user_reward)
    # print(batch)
    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    action_mask = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, r, mask, action_mask)


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
    parser.add_argument("--user-reward", action="store_true")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs',
                        f'{parser.parse_args().config_name}.json')
    seed = parser.parse_args().seed
    mode = parser.parse_args().mode
    save_eval = parser.parse_args().save_eval_dials
    use_user_reward = parser.parse_args().user_reward

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)

    args = [('model', 'seed', seed)] if seed is not None else list()

    environment_config = load_config_file(path)
    save_config(vars(parser.parse_args()),
                environment_config, config_save_path)

    conf = get_config(path, args)
    seed = conf['model']['seed']
    logging.info('Train seed is ' + str(seed))
    set_seed(seed)

    policy_sys = PPO(True, seed=conf['model']['seed'],
                     vectorizer=conf['vectorizer_sys_activated'])

    # Load model
    if conf['model']['use_pretrained_initialisation']:
        logging.info("Loading supervised model checkpoint.")
        policy_sys.load_from_pretrained(
            conf['model'].get('pretrained_load_path', ""))
    elif conf['model']['load_path']:
        try:
            policy_sys.load(conf['model']['load_path'])
        except Exception as e:
            logging.info(f"Could not load a policy: {e}")
    else:
        logging.info("Policy initialised from scratch")

    log_start_args(conf)
    logging.info(f"New episodes per epoch: {conf['model']['num_train_dialogues']}")

    env, sess = env_config(conf, policy_sys)

    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path

    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    eval_dict = eval_policy(conf, policy_sys, env, sess,
                            save_eval, log_save_path)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")

    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], 0)
    best_complete_rate = eval_dict['complete_rate']
    best_success_rate = eval_dict['success_rate_strict']
    best_return = eval_dict['avg_return']

    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    for i in range(conf['model']['epoch']):
        idx = i + 1
        # print("Epoch :{}".format(str(idx)))
        update(env, policy_sys, conf['model']['num_train_dialogues'], idx, conf['model']['process_num'], seed=seed, user_reward=use_user_reward)

        if idx % conf['model']['eval_frequency'] == 0 and idx != 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(f"Evaluating after Dialogues: {idx * conf['model']['num_train_dialogues']} - {time_now}" + '-' * 60)

            eval_dict = eval_policy(
                conf, policy_sys, env, sess, save_eval, log_save_path)

            best_complete_rate, best_success_rate, best_return = \
                save_best(policy_sys, best_complete_rate, best_success_rate, best_return,
                          eval_dict["complete_rate"], eval_dict["success_rate_strict"],
                          eval_dict["avg_return"], save_path)
            policy_sys.save(save_path, "last")
            for key in eval_dict:
                tb_writer.add_scalar(key, eval_dict[key], idx * conf['model']['num_train_dialogues'])
    logging.info("End of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))
