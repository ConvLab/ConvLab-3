import argparse
import os
import torch
import logging
import json
import sys

from torch import nn

from convlab.policy.mle.loader import PolicyDataVectorizer
from convlab.util.custom_util import set_seed, init_logging, save_config
from convlab.util.train_util import to_device
from convlab.policy.rlmodule import MultiDiscretePolicy
from convlab.policy.vector.vector_binary import VectorBinary
from convlab.policy.vector.vector_binary_fuzzy import VectorBinaryFuzzy

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE_Trainer_Abstract():
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        self.policy = None
        self.policy_optim = None

    def _init_data(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('validation', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()

    def policy_loop(self, data):

        s, target_a, mask = to_device(data)
        a_weights = self.policy(s)

        loss_a = self.multi_entropy_loss(a_weights + mask, target_a)
        return loss_a

    def imitating(self):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()

        self.policy.eval()

    def validate(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        average_actions, average_target_actions, counter = 0, 0, 0
        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_valid):
            counter += 1
            s, target_a, m = to_device(data)
            a_weights = self.policy(s)
            a_weights += m
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

            average_actions += a.float().sum(dim=-1).mean()
            average_target_actions += target_a.float().sum(dim=-1).mean()

        logging.info(f"Average actions: {average_actions / counter}")
        logging.info(f"Average target actions: {average_target_actions / counter}")
        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        return prec, rec, F1

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.state_dict(), directory + '/supervised.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))


class MLE_Trainer(MLE_Trainer_Abstract):
    def __init__(self, manager, vector, cfg):
        self._init_data(manager, cfg)

        # override the loss defined in the MLE_Trainer_Abstract to support pos_weight
        pos_weight = cfg['pos_weight'] * torch.ones(vector.da_dim).to(device=DEVICE)
        self.multi_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.da_dim).to(device=DEVICE)
        self.policy.eval()
        self.policy_optim = torch.optim.RMSprop(self.policy.parameters(), lr=cfg['lr_supervised'],
                                                weight_decay=cfg['weight_decay'])


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="multiwoz21")
    parser.add_argument("--use_masking", action='store_true')

    parser.add_argument("--dst", type=str, default=None)
    parser.add_argument("--dst_args", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = arg_parser()

    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, 'config.json'), 'r') as f:
        cfg = json.load(f)

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), "info")
    save_config(vars(args), cfg, config_save_path)

    set_seed(args.seed)
    logging.info(f"Seed used: {args.seed}")

    if args.dst is None:
        vector = VectorBinary(dataset_name=args.dataset_name, use_masking=args.use_masking)
        dst = None
    elif args.dst == "setsumbt":
        dst_args = [arg.split('=', 1) for arg in args.dst_args.split(', ')
                    if '=' in arg] if args.dst_args is not None else []
        dst_args = {key: eval(value) for key, value in dst_args}
        from convlab.dst.setsumbt import SetSUMBTTracker
        dst = SetSUMBTTracker(**dst_args)
        if dst.return_confidence_scores:
            from convlab.policy.vector.vector_uncertainty import VectorUncertainty
            vector = VectorUncertainty(dataset_name=args.dataset_name, use_masking=args.use_masking,
                                       manually_add_entity_names=False,
                                       use_confidence_scores=dst.return_confidence_scores,
                                       confidence_thresholds=dst.confidence_thresholds,
                                       use_state_total_uncertainty=dst.return_belief_state_entropy,
                                       use_state_knowledge_uncertainty=dst.return_belief_state_mutual_info)
        else:
            vector = VectorBinary(dataset_name=args.dataset_name, use_masking=args.use_masking)
    elif args.dst == "trippy":
        dst_args = [arg.split('=', 1) for arg in args.dst_args.split(', ')
                    if '=' in arg] if args.dst_args is not None else []
        dst_args = {key: eval(value) for key, value in dst_args}
        from convlab.dst.trippy import TRIPPY
        dst = TRIPPY(**dst_args)
        vector = VectorBinaryFuzzy(dataset_name=args.dataset_name, use_masking=args.use_masking)
    else:
        raise NameError(f"Tracker: {args.dst} not implemented.")
    manager = PolicyDataVectorizer(dataset_name=args.dataset_name, vector=vector, dst=dst)
    agent = MLE_Trainer(manager, vector, cfg)

    logging.info('Start training')

    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    precision = 0
    recall = 0
    f1 = 0

    for e in range(cfg['epoch']):
        agent.imitating()
        logging.info(f"Epoch: {e}")

        if e % args.eval_freq == 0 and e != 0:
            precision, recall, f1 = agent.validate()

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1: {f1}")

        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if f1 > best_f1:
            best_f1 = f1
            agent.save(save_path, e)
        logging.info(f"Best Precision: {best_precision}")
        logging.info(f"Best Recall: {best_recall}")
        logging.info(f"Best F1: {best_f1}")
