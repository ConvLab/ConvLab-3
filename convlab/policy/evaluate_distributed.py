# -*- coding: utf-8 -*-

import random
import torch
import numpy as np

from convlab.policy.rlmodule import Memory_evaluator
from torch import multiprocessing as mp


def sampler(pid, queue, evt, sess, seed_range, goals):
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
    buff = Memory_evaluator()

    for seed in seed_range:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        goal = goals.pop()
        sess.init_session(goal=goal)
        sys_response = '' if sess.sys_agent.nlg is not None else []
        sys_response = [] if sess.sys_agent.return_semantic_acts else sys_response
        total_return_success = 0.0
        total_return_complete = 0.0
        turns = 0
        complete = 0
        success = 0
        success_strict = 0
        avg_actions = 0
        book = 0
        inform = 0
        request = 0
        select = 0
        offer = 0
        recommend = 0
        task_success = {}

        for i in range(40):
            # TODO: I think the reward here is also from user simulator and not evaluator, check for task-success if yes
            sys_response, user_response, session_over, reward = sess.next_turn(
                sys_response)

            turns += 1
            total_return_success += sess.evaluator.get_reward(terminated=session_over)
            total_return_complete += sess.user_agent.policy.policy.get_reward()
            acts = sess.sys_agent.dst.state['system_action']
            avg_actions += len(acts)

            for intent, domain, _, _ in acts:
                if intent.lower() == 'book':
                    book += 1
                if intent.lower() == 'inform':
                    inform += 1
                if intent.lower() == 'request':
                    request += 1
                if intent.lower() == 'select':
                    select += 1
                if intent.lower() == 'offerbook':
                    offer += 1
                if intent.lower() == 'recommend':
                    recommend += 1

            if session_over is True:
                success = sess.evaluator.task_success()
                complete = sess.evaluator.complete
                success = sess.evaluator.success
                success_strict = sess.evaluator.success_strict
                break

        for key in sess.evaluator.goal:
            if key not in task_success:
                task_success[key] = []
            task_success[key].append(success_strict)

        buff.push(complete, success, success_strict, total_return_complete, total_return_success, turns, avg_actions / turns,
                  task_success, book, inform, request, select, offer, recommend)

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(sess, seedrange, process_num, goals):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
        :param env:
        :param policy:
    :param batchsz:
        :param process_num:
    :return: batch
    """

    num_seeds = len(seedrange)
    num_seeds_per_thread = np.ceil(num_seeds / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (
            i, queue, evt, sess, seedrange[i * num_seeds_per_thread: (i+1) * num_seeds_per_thread],
            goals[i * num_seeds_per_thread: (i+1) * num_seeds_per_thread])
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


def evaluate_distributed(sess, seed_range, process_num, goals):

    batch = sample(sess, seed_range, process_num, goals)
    return batch.complete, batch.success, batch.success_strict, batch.total_return_success, batch.turns, \
           batch.avg_actions, batch.task_success, np.average(batch.book_actions), np.average(batch.inform_actions), \
           np.average(batch.request_actions), np.average(batch.select_actions), np.average(batch.offer_actions), \
           np.average(batch.recommend_actions)


if __name__ == "__main__":
    pass
