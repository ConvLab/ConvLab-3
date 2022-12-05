# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: lubis@hhu.de
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch import multiprocessing as mp
from convlab.dialog_agent.agent import PipelineAgent
from convlab.dialog_agent.env import Environment
from convlab.nlu.svm.multiwoz import SVMNLU
from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.dst.rule.multiwoz import RuleDST
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.policy.ppo import PPO
from convlab.policy.lava.multiwoz import LAVA
from convlab.policy.rlmodule import Memory_LAVA, Transition_LAVA
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.util.analysis_tool.analyzer import Analyzer
from argparse import ArgumentParser
import torch as th
from tqdm import tqdm
import pdb
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    th.manual_seed(r_seed)

def mpsampler(pid, queue, evt, env, policy, batchsz):
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
            a = policy.predict(s)
            logprobs = policy.logprobs

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            # next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s, a, logprobs, r, next_s, mask)

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

def mpsample(env, policy, batchsz, process_num):
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
        ForkedPdb().set_trace()
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

def interact(env, policy, episode_num):
    """
    This is a sampler function
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory_LAVA()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    #while sampled_num < batchsz:
    for i in tqdm(range(episode_num), desc="sampling episode for this epoch"):
        # for each trajectory, we reset the env and get initial state
        s = env.reset()


        for t in range(traj_len):

            # [s_dim] => [a_dim]
            a = policy.predict(s)
            logprobs = th.stack(policy.logprobs)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            # next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s, a, logprobs, r, next_s, mask)

            # update per step
            real_traj_len = t

            if done:
                buff.push_episode()
                batch = Transition_LAVA(*zip(*buff.memory[-1]))
                r = torch.from_numpy(np.stack(batch.reward).astype("float")).to(device=DEVICE)
                policy.update(r, th.stack(batch.logprobs))

                break


        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length
    return buff


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--episode_num", type=int, default=100, help="number of episode sampled per epoch")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
    #parser.add_argument("--process_num", type=int, default=3, help="number of processes of trajactory sampling")
    parser.add_argument("--lava_model_type", type=str, default="actz_cat", help="which LAVA variant to choose")
    parser.add_argument("--lex_type", type=str, default="default", help="which lexicalization function to choose, default or augpt")
    args = parser.parse_args()

    ##agent##
    sys_nlu = BERTNLU()
    sys_dst = RuleDST()
    model_path = {}
    model_path["actz_gauss"] = "2020-05-12-15-29-19-actz_gauss/rl-2020-05-18-11-53-22"
    model_path["actz_cat"] = "2020-05-12-14-51-49-actz_cat/rl-2020-05-18-10-50-48"
    model_path["mt_gauss"] = "2020-04-06-11-36-48-mt_gauss/rl-2020-04-08-09-42-59"
    model_path["mt_cat"] = "2020-04-07-08-43-50-mt_cat/rl-2020-04-08-09-50-11"
    model_path["ptS_cat"] = "2020-02-26-18-11-37-sl_cat_ae/rgft-2020-10-13-15-07-56/rl-2020-10-13-16-40-58"
    model_path["ptA_cat"] = "2020-02-26-18-11-37-sl_cat_ae/rgft-2020-03-06-16-23-12/rl-2020-03-17-22-56-19"
    model_path["ptS_gauss"] = "2020-02-28-16-49-48-sl_gauss_ae/rgft-2020-03-11-16-20-07/rl-2020-03-18-15-31-56"
    model_path["ptA_gauss"] = "2020-02-28-16-49-48-sl_gauss_ae/rgft-2020-04-20-12-38-57/rl-2020-04-21-08-34-05"
    model_path["actz_cat_long_context"] = "2021-02-11-17-55-19-actz_cat/rl-2021-02-11-21-35-23"
    model_path["actz_e2e_cat"] = "2020-06-26-11-23-54-actz_e2e_cat/rl-2020-06-26-13-40-58"
    model_path["baseline"] = "2020-02-10-15-48-40-sl_cat/rl-2020-02-11-15-23-19" #larl
    # lava_path = "/gpfs/project/lubis/public_code/LAVA/experiments_woz/sys_config_log_model/{}/reward_best.model".format(model_path[args.lava_model_type])
    lava_path = "/gpfs/project/lubis/NeuralDialog-LaRL/experiments_woz/sys_config_log_model/{}/reward_best.model".format(model_path[args.lava_model_type])
    #lava_path = "/home/lubis/gpfs/project/lubis/neuraldialog-larl/experiments_woz/sys_config_log_model/{}/reward_best.model".format(model_path[args.lava_model_type])
    print(lava_path)
    #TODO change rl_config
    sys_policy = LAVA(lava_path, args.lex_type, is_train=True)
    sys_nlg=None
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    ##USER##
    # BERT nlu trained on sys utterance
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                       model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip')
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr')
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    evaluator = MultiWozEvaluator()
    env = Environment(None, user_agent, sys_nlu, sys_dst, evaluator)

    seed = 0
    set_seed(seed)

    print("analyzing initial performance")
    model_name = 'bertnlu_rule-lava-{}+RL-{}_lexicalizer-ep'.format(args.lava_model_type, args.lex_type)
    try:
        model_init_dict = sys_policy.model.state_dict()['c2z.p_h.weight'].clone()
    except:
        model_init_dict = sys_policy.model.state_dict()['c2z.mu.weight'].clone()
    sys_policy.model.eval()
    complete, success, prec, recall, f1, match, avg_turn = analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=model_name+"0", total_dialog=100)
    sys_policy.model.train()

    best_success = success
    best_epoch = 0
    buff = Memory_LAVA()
    for i in tqdm(range(args.epoch), desc="RL epoch"):
        new_buff = interact(env, sys_policy, args.episode_num)
        buff.append(new_buff)
        if i % 10 == 0 and i != 0:
            sys_policy.model.eval()
            complete, success, prec, recall, f1, match, avg_turn = analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=model_name + str(i), total_dialog=100)
            if success > best_success:
                best_success = success
                best_epoch = i
                print("New model saved at epoch {} with {} success rate!".format(i, best_success))
                #pdb.set_trace()
                try:
                    model_current_dict = sys_policy.model.state_dict()['c2z.p_h.weight'].clone()
                except:
                    model_current_dict = sys_policy.model.state_dict()['c2z.mu.weight'].clone()
                print(model_current_dict - model_init_dict)
                sys_policy.save()
            analyzer.sample_dialog(sys_agent)
            sys_policy.model.train()
    
    #print("analyzing final performance")
    #sys_policy.model.eval()
    #analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=model_name + str(args.epoch), total_dialog=100)
