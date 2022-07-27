from torch import multiprocessing as mp
from copy import deepcopy

import logging
import torch
import time
from convlab.util.custom_util import set_seed

torch.multiprocessing.set_sharing_strategy('file_system')

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


# we use a queue for every process to guarantee reproducibility
# queues are used for job submission, while episode queues are used for pushing dialogues inside
def get_queues(train_processes):
    queues = []
    episode_queues = []
    for p in range(train_processes):
        queues.append(mp.SimpleQueue())
        episode_queues.append(mp.SimpleQueue())

    return queues, episode_queues


# this is our target function for the processes
def create_episodes_process(do_queue, put_queue, environment, policy, seed, metric_queue):
    traj_len = 40
    set_seed(seed)

    while True:
        if not do_queue.empty():
            item = do_queue.get()
            if item == 'stop':
                print("Got stop signal.")
                break
            else:
                s = environment.reset(item)
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
                        metric_queue.put({"success": environment.evaluator.success_strict, "return": rl_return,
                                          "avg_actions": torch.stack(action_list).sum(dim=-1).mean().item(),
                                          "turns": t, "goal": item.domain_goals})
                        put_queue.put((description_idx_list, action_list, reward_list, small_act_list, mu_list,
                                       action_mask_list, critic_value_list, description_idx_list, value_list,
                                       current_domain_mask, non_current_domain_mask))
                        break


def start_processes(train_processes, queues, episode_queues, env, policy_sys, seed, metric_queue):
    logging.info("Spawning processes..")
    processes = []
    for i in range(train_processes):
        process_args = (queues[i], episode_queues[i], env, policy_sys, seed, metric_queue)
        p = mp.Process(target=create_episodes_process, args=process_args)
        processes.append(p)
    for b, p in enumerate(processes):
        p.daemon = True
        p.start()
        logging.info(f"Started process {b}")
    return processes


def terminate_processes(processes, queues):
    # kill processes properly
    logging.info("Terminating processes..")
    for b, p in enumerate(processes):
        queues[b].put('stop')
    time.sleep(2)
    for b, p in enumerate(processes):
        p.terminate()
        logging.info(f"Terminated process {b}")


def submit_jobs(num_jobs, queues, episode_queues, train_processes, memory, goals, metric_queue):
    # first create goals with global environment and put them into queue.
    # If every environment process would do that itself, it could happen that environment 1 creates 24 dialogues in
    # one run and 25 in another run (for two processes and 50 jobs for instance)
    metrics = []
    for job in range(num_jobs):
        if goals:
            goal = goals.pop()
            queues[job % train_processes].put(goal)
    time_now = time.time()
    collected_dialogues = 0
    episode_list = []
    for b in range(train_processes):
        episode_list.append([])
    # we need to have a dialogue list for every process, otherwise it could happen that the order in which dialogues
    # are pushed into the list is different for different runs
    # in the end the dialogue lists are just appended basically instead of being possibly mixed
    while collected_dialogues != num_jobs:
        for b in range(train_processes):
            if not episode_queues[b].empty():
                metrics.append(metric_queue.get())
                dialogue = episode_queues[b].get()
                dialogue_ = deepcopy(dialogue)
                episode_list[b].append(dialogue_)
                del dialogue
                collected_dialogues += 1
    for b in range(train_processes):
        for dialogue in episode_list[b]:
            memory.update_episode(*dialogue)
    return time_now, metrics
