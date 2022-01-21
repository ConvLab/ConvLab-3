# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""

import sys
import os
import logging
import time
import numpy as np
import torch

from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import Memory
from torch import multiprocessing as mp
from argparse import ArgumentParser
from convlab2.util.custom_util import set_seed, init_logging, save_config, move_finished_training, create_env, \
    eval_policy, log_start_args, save_best

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def sampler(pid, queue, evt, env, policy, batchsz):
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

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec, action_mask = policy.vector.state_vectorize(s)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)

            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

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


def sample(env, policy, batchsz, process_num):
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
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
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
        process_args = (i, queue, evt, env, policy, process_batchsz)
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


def update(env, policy, batchsz, epoch, process_num, only_critic=False):

    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    action_mask = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, r, mask,
                  action_mask, only_critic=only_critic)


if __name__ == '__main__':

    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    begin_time = datetime.now()
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="convlab2/policy/ppo/best_supervised",
                        help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1000,
                        help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=200,
                        help="number of epochs to train")
    parser.add_argument("--eval_frequency", type=int,
                        default=5, help="evaluation frequency")
    parser.add_argument("--process_num", type=int, default=6,
                        help="number of processes of trajactory sampling")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the policy parameter initialization")
    parser.add_argument("--use_masking", action='store_true',
                        help="Use action masking for PPO")
    parser.add_argument("--num_eval_dialogues", type=int,
                        default=400, help="Number of dialogues for evaluation")

    # Tracker args
    parser.add_argument("--use_setsumbt_tracker", action='store_true',
                        help="whether to use setsumbt tracker during RL")
    parser.add_argument("--sys_semantic_to_usr", action='store_true',
                        help="whether to use send semnatic acts directly from system to simulator")
    parser.add_argument("--use_bertnlu_rule_tracker", action='store_true',
                        help="whether to use bertnlu + rule tracker during RL")
    parser.add_argument("--setsumbt_path", type=str,
                        default='/gpfs/project/niekerk/results/nbt/convlab_setsumbt_acts_distilled')
    parser.add_argument("--nlu_model_path", type=str,
                        default='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
    parser.add_argument("--use_confidence_scores", action='store_true',
                        help="whether to use belief probabilities in the state")
    parser.add_argument("--use_state_entropy", action='store_true',
                        help="whether to use entropy of state distributions in the state")
    parser.add_argument("--use_state_mutual_info", action='store_true',
                        help="whether to use mutual information of state distributions in the state")
    parser.add_argument("--user_label_noise", type=float,
                        default=0.0, help="User simulator label noise level")
    parser.add_argument("--user_text_noise", type=float,
                        default=0.0, help="User simulator text noise level")

    args = parser.parse_args()

    if args.use_setsumbt_tracker:
        args.use_bertnlu_rule_tracker = False
    if args.use_bertnlu_rule_tracker:
        args.use_belief_probs = False

    set_seed(args.seed)

    policy_sys = PPO(True, seed=args.seed, use_action_mask=args.use_masking, shrink=False,
                     use_entropy=args.use_state_entropy, use_mutual_info=args.use_state_mutual_info,
                     use_confidence_scores=args.use_confidence_scores)
    policy_sys.load(args.load_path)

    logger, tb_writer, current_time, save_path, config_save_path, dir_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)))

    log_start_args(args)
    logging.info(f"New episodes per epoch: {args.batchsz}")

    env, sess = create_env(args, policy_sys)

    if args.use_confidence_scores:
        policy_sys.vector.setup_uncertain_query(env.sys_dst.thresholds)

    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path

    save_config(vars(args), policy_sys.cfg, config_save_path)

    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    complete_rate, success_rate, avg_return, turns, avg_actions = eval_policy(
        args, policy_sys, env, sess)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")

    tb_writer.add_scalar('complete_rate', complete_rate, 0)
    tb_writer.add_scalar('success_rate', success_rate, 0)
    tb_writer.add_scalar('avg_return', avg_return, 0)
    tb_writer.add_scalar('turns', turns, 0)
    tb_writer.add_scalar('avg_actions', avg_actions, 0)

    best_complete_rate = complete_rate
    best_success_rate = success_rate

    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    for i in range(conf['model']['epoch']):
        idx = i + 1
        update(env, policy_sys, conf['model']['batchsz'],
               idx, conf['model']['process_num'])

        if idx % conf['model']['eval_frequency'] == 0 and idx != 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(f"Evaluating at Epoch: {idx} - {time_now}" + '-'*60)

            complete_rate, success_rate, avg_return, turns, avg_actions = eval_policy(
                conf, policy_sys, env, sess)
            best_complete_rate, best_success_rate = \
                save_best(policy_sys, best_complete_rate, best_success_rate,
                          complete_rate, success_rate, save_path)

            tb_writer.add_scalar('complete_rate', complete_rate,
                                 idx * conf['model']['batchsz'])
            tb_writer.add_scalar('success_rate', success_rate,
                                 idx * conf['model']['batchsz'])
            tb_writer.add_scalar('avg_return', avg_return,
                                 idx * conf['model']['batchsz'])
            tb_writer.add_scalar('turns', turns, idx *
                                 conf['model']['batchsz'])
            tb_writer.add_scalar('avg_actions', avg_actions,
                                 idx * conf['model']['batchsz'])

    logging.info("End of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))
