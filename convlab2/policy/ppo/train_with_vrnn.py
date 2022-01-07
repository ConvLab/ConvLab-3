# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""

import sys, os, logging, random, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from tensorboardX import SummaryWriter
from convlab2.dialog_agent.session import BiSession
from convlab2.vrnn_semantic.model import VRNN


from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import Memory_vrnn
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser
from convlab2.dialog_agent.session import BiSession
#from convlab2.util.train_util import save_to_bucket, init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
voc_file_shrinked = os.path.join(root_dir, 'data/multiwoz/sys_da_voc_shrinked.txt')
voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
VECTOR = MultiWozVector(voc_file_shrinked, voc_opp_file, shrink=True)

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def save_log_to_bucket(policy_sys, bucket_dir='logs_PPO_masking_no_pretraining'):
    try:
        save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/{policy_sys.save_name}/log.txt',
                       os.path.join(policy_sys.log_dir, f'log_{policy_sys.current_time}.txt'))
        for file in os.listdir(policy_sys.log_dir):
            if 'events.out' in file:
                save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/{policy_sys.save_name}/{file}',
                               os.path.join(policy_sys.log_dir, file))
    except:
        logging.info('Could not save to bucket.')


def save_best_policy(policy, rate, best_rate, success=True, bucket_dir='logs_PPO_masking_no_pretraining'):
    #try block is only for local testing, when we can not access the gcloud storage
    try:
        if rate > best_rate:
            if success:
                policy.save(policy.save_dir, best_success_rate=True)
                save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/'
                                             f'{policy_sys.save_name}/best_success_rate_ppo.val.mdl',
                               os.path.join(policy.save_dir, policy.current_time + '_best_success_rate_ppo.val.mdl'))
                save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/'
                                             f'{policy_sys.save_name}/best_success_rate_ppo.pol.mdl',
                               os.path.join(policy.save_dir, policy.current_time + '_best_success_rate_ppo.pol.mdl'))
            else:
                policy.save(policy.save_dir, best_complete_rate=True)
                save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/'
                                             f'{policy_sys.save_name}/best_complete_rate_ppo.val.mdl',
                               os.path.join(policy.save_dir, policy.current_time + '_best_complete_rate_ppo.val.mdl'))
                save_to_bucket('geishauser', f'convlab_experiments/{bucket_dir}/'
                                             f'{policy_sys.save_name}/best_complete_rate_ppo.pol.mdl',
                               os.path.join(policy.save_dir, policy.current_time + '_best_complete_rate_ppo.pol.mdl'))
            return rate
        else:
            return best_rate
    except:
        return best_rate


def evaluate(dataset_name, load_path=None, calculate_reward=False, policy_sys=None, counter=0, writer=None, evaluator_reward=False):
    seed = 20190827
    random.seed(seed)
    np.random.seed(seed)

    if dataset_name == 'MultiWOZ':
        dst_sys = RuleDST()

        from convlab2.policy.ppo import PPO
        if policy_sys is None:
            if load_path:
                policy_sys = PPO(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PPO.from_pretrained

        dst_usr = None

        policy_usr = RulePolicy(character='usr')
        simulator = PipelineAgent(None, None, policy_usr, None, 'user')

        env = Environment(None, simulator, None, dst_sys)

        agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

        evaluator = MultiWozEvaluator()

        sess = BiSession(agent_sys, simulator, None, evaluator)

        seeds = random.randrange(400, 1000000)

        actions = 0.0
        turn_counter = 0.0

        task_success = {'All_user_sim': [], 'All_evaluator': [], 'total_return': [], 'turns': []}
        for seed in range(seeds - 400, seeds):
            random.seed(seed)
            np.random.seed(seed)
            sess.init_session()
            sys_response = []

            total_return = 0.0
            turns = 0
            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
                actions += len(sys_response)
                turn_counter += 1
                turns = i
                if evaluator_reward:
                    total_return += reward
                else:
                    total_return += simulator.policy.policy.get_reward()

                if session_over is True:
                    task_succ = sess.evaluator.task_success()

                    break
            else:
                task_succ = 0

            for key in sess.evaluator.goal:
                if key not in task_success:
                    task_success[key] = []
                else:
                    task_success[key].append(task_succ)

            task_success['All_user_sim'].append(int(simulator.policy.policy.goal.task_complete()))
            task_success['All_evaluator'].append(task_succ)
            task_success['total_return'].append(total_return)
            task_success['turns'].append(turns)

        with open(os.path.join(policy_sys.log_dir, f'log_{policy_sys.current_time}.txt'), 'a') as log_file:

            for key in task_success:
                logging.info(
                    f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}')

                log_file.write(f'{key} {len(task_success[key])} {np.average(task_success[key]) if len(task_success[key]) > 0 else 0}\n')

            logging.info(f"Average number of actions per turn: {actions/turn_counter}")
            #log_file.write(f"Average number of actions per turn: {actions/turn_counter}\n")

        if writer is not None:
            writer.write_summary(np.average(task_success['All_user_sim']), np.average(task_success['All_evaluator']),
                                 np.average(task_success['turns']), np.average(task_success['total_return']), counter)

        return np.average(task_success['All_user_sim']), np.average(task_success['All_evaluator'])
    else:
        raise Exception("currently supported dataset: MultiWOZ")


def sampler(pid, queue, evt, env, policy, batchsz, vrnn_model=None, mc_samples=1.0):
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
    buff = Memory_vrnn()

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

        user_act_list, sys_act_list, s_vec_list, next_s_vec_list, action_list, reward_list, trajectory_list, \
        mask_list, action_mask_list = [], [], [], [], [], [], [], [], []

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec, action_mask = policy.vector.state_vectorize(s, output_mask=True)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)

            a = policy.predict(s)

            user_act_list.append(policy.vector.retrieve_user_action(s))
            sys_act_list.append(VECTOR.action_vectorize(a))

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec, next_action_mask = policy.vector.state_vectorize(next_s, output_mask=True)
            next_s_vec = torch.Tensor(next_s_vec)

            s_vec_list.append(s_vec.numpy())
            action_list.append(policy.vector.action_vectorize(a))
            reward_list.append(r)
            next_s_vec_list.append(next_s_vec.numpy())
            mask_list.append(mask)
            action_mask_list.append(action_mask.numpy())

            # save to queue
            #buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask, action_mask.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        if vrnn_model:
            crossentropy_loss = compute_vrnn_reward(mc_samples, sys_act_list, user_act_list, vrnn_model)

            for i in range(len(user_act_list)):
                buff.push(s_vec_list[i], action_list[i], reward_list[i], next_s_vec_list[i], mask_list[i],
                          action_mask_list[i], crossentropy_loss[i])

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def compute_vrnn_reward(mc_samples, sys_act_list, user_act_list, vrnn_model):
    _, _, kld_loss, nll_loss = \
        vrnn_model(torch.Tensor([user_act_list]).repeat(mc_samples, 1, 1).to(DEVICE),
                   torch.Tensor([sys_act_list]).repeat(mc_samples, 1, 1).to(DEVICE), None,
                   torch.Tensor([len(user_act_list)]).repeat(mc_samples).to(DEVICE))

    crossentropy_loss = -torch.stack(kld_loss) - torch.stack(nll_loss)

    # normalize reward, should maybe do it across episodes
    #crossentropy_loss = (crossentropy_loss - crossentropy_loss.mean()) / (crossentropy_loss.std() + 0.00001)
    crossentropy_loss = crossentropy_loss.detach()
    return crossentropy_loss


def sample(env, policy, batchsz, process_num, vrnn_model=None, mc_samples=1):
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
        process_args = (i, queue, evt, env, policy, process_batchsz, vrnn_model, mc_samples)
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


def update(env, policy, batchsz, epoch, process_num, only_critic=False, vrnn_model=None, mc_samples=1, ce_weighting=1.0):

    # sample data asynchronously
    if vrnn_model is not None:
        batch = sample(env, policy, batchsz, process_num, vrnn_model, mc_samples)
    else:
        batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    action_mask = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
    vrnn_reward = torch.Tensor(np.stack(batch.vrnn_reward)).to(device=DEVICE)
    batchsz_real = s.size(0)

    r_new = r + ce_weighting * (vrnn_reward - vrnn_reward.mean()) / (vrnn_reward.std() + 0.00001)

    policy.update(epoch, batchsz_real, s, a, r_new, mask, action_mask, only_critic=only_critic)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1000, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the policy parameter initialization")
    parser.add_argument("--action_mask", type=bool, default=False, help="Use action masking for PPO")
    parser.add_argument("--vrnn_path", type=str, default="")
    parser.add_argument("--mc_num", type=int, default=1, help="How many MonteCarlo samples for VRNN")
    parser.add_argument("--ce_weight", type=float, default=0.025, help="Weight of cross-entropy reward")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    if args.action_mask:
        args.action_mask = True
    else:
        args.action_mask = False

    print("ACTION MASK: ", args.action_mask)

    bucket_dir = 'logs_PPO_masking_no_pretraining'
    if args.load_path and args.action_mask:
        bucket_dir = 'logs_PPO_masking_pretraining'
    elif args.load_path and not args.action_mask:
        bucket_dir = 'logs_PPO_pretraining'
    elif not args.load_path and not args.action_mask:
        bucket_dir = 'logs_PPO_no_pretraining'

    bucket_dir += "_discrete"

    # simple rule DST
    dst_sys = RuleDST()

    policy_sys = PPO(True, seed=args.seed, use_action_mask=args.action_mask, shrink=False)
    policy_sys.load(args.load_path)

    if args.vrnn_path:
        tb_writer = SummaryWriter(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               f'TB_summary/{policy_sys.current_time}_vrnn_normalized'))
    else:
        tb_writer = SummaryWriter(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               f'TB_summary/{policy_sys.current_time}'))

    if args.vrnn_path:
        logging.info("We use VRNN reward signal")
        with torch.no_grad():
            vrnn = VRNN(300, 300, 300, 1, 30).to(DEVICE)
            vrnn.load_state_dict(torch.load(args.vrnn_path, map_location=DEVICE))
            vrnn.eval()

    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = RulePolicy(character='usr')

    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    evaluator = MultiWozEvaluator()
    env = Environment(None, simulator, None, dst_sys, evaluator=evaluator)

    logging.info("Start of Training: " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    best_complete_rate = 0.0
    best_success_rate = 0.0

    #if args.load_path:
    #    for i in range(20):
    #        print("Updating only critic")
    #        update(env, policy_sys, args.batchsz, i, args.process_num, only_critic=True)

    for i in range(args.epoch):

        if args.vrnn_path:
            update(env, policy_sys, args.batchsz, i, args.process_num, vrnn_model=vrnn, mc_samples=args.mc_num,
                            ce_weighting=args.ce_weight)
        else:
            update(env, policy_sys, args.batchsz, i, args.process_num)

        logging.info(f"Evaluating at Epoch: {i} " + '-' * 80)
        with open(os.path.join(policy_sys.log_dir, f'log_{policy_sys.current_time}.txt'), 'a') as log_file:
            #log_file.write(f"Evaluating at Epoch: {i} " + '-' * 80 + "\n")
            pass

        policy_sys.is_train = False
        complete_rate, success_rate = evaluate('MultiWOZ', policy_sys=policy_sys, counter=i * 1000, writer=None)
        tb_writer.add_scalar('complete_rate', complete_rate, i * args.batchsz)
        tb_writer.add_scalar('success_rate', success_rate, i * args.batchsz)
        policy_sys.is_train = True
        save_log_to_bucket(policy_sys, bucket_dir=bucket_dir)

        best_complete_rate = save_best_policy(policy_sys, complete_rate, best_complete_rate, success=False, bucket_dir=bucket_dir)
        best_success_rate = save_best_policy(policy_sys, success_rate, best_success_rate, success=True, bucket_dir=bucket_dir)

    logging.info("End of Training: " + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
