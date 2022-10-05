import argparse
import os
import torch
import logging
import json
import sys

from torch import optim
from copy import deepcopy
from convlab.policy.vtrace_DPT.supervised.loader import PolicyDataVectorizer
from convlab.util.custom_util import set_seed, init_logging, save_config
from convlab.util.train_util import to_device
from convlab.policy.vtrace_DPT.transformer_model.EncoderDecoder import EncoderDecoder
from convlab.policy.vector.vector_nodes import VectorNodes

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE_Trainer:
    def __init__(self, manager, cfg, policy):
        self.start_policy = deepcopy(policy)
        self.policy = policy
        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=cfg['supervised_lr'])
        self.entropy_weight = cfg['entropy_weight']
        self.regularization_weight = cfg['regularization_weight']
        self._init_data(manager, cfg)

    def _init_data(self, manager, cfg):
        multiwoz_like = cfg['multiwoz_like']
        self.data_train, self.max_length_train, self.small_act_train, self.descriptions_train, self.values_train, \
            self.kg_train = manager.create_dataset('train', cfg['batchsz'], self.policy, multiwoz_like)
        self.data_valid, self.max_length_valid, self.small_act_valid, self.descriptions_valid, self.values_valid, \
            self.kg_valid = manager.create_dataset('validation', cfg['batchsz'], self.policy, multiwoz_like)
        self.data_test, self.max_length_test, self.small_act_test, self.descriptions_test, self.values_test, \
            self.kg_test = manager.create_dataset('test', cfg['batchsz'], self.policy, multiwoz_like)
        self.save_dir = cfg['save_dir']

    def policy_loop(self, data):

        actions, action_masks, current_domain_mask, non_current_domain_mask, indices = to_device(data)

        small_act_batch = [self.small_act_train[i].to(DEVICE) for i in indices]
        description_batch = [self.descriptions_train[i].to(DEVICE) for i in indices]
        value_batch = [self.values_train[i].to(DEVICE) for i in indices]

        log_prob, entropy = self.policy.get_log_prob(actions, action_masks, self.max_length_train, small_act_batch,
                                 current_domain_mask, non_current_domain_mask,
                                 description_batch, value_batch)
        loss_a = -1 * log_prob.mean()

        weight_loss = self.weight_loss()

        return loss_a, -entropy, weight_loss

    def weight_loss(self):

        loss = 0
        num_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        for paramA, paramB in zip(self.policy.parameters(), self.start_policy.parameters()):
            loss += torch.sum(torch.abs(paramA - paramB.detach()))
        return loss / num_params

    def imitating(self):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a, entropy_loss, weight_loss = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a = loss_a + self.entropy_weight * entropy_loss + self.regularization_weight * weight_loss

            if i % 20 == 0 and i != 0:
                print("LOSS:", a_loss / 20.0)
                a_loss = 0
            loss_a.backward()
            for p in self.policy.parameters():
                if p.grad is not None:
                    p.grad[p.grad != p.grad] = 0.0
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
            target_a, action_masks, current_domain_mask, non_current_domain_mask, indices = to_device(data)

            kg_batch = [self.kg_valid[i] for i in indices]
            a = torch.stack([self.policy.select_action([kg]) for kg in kg_batch])

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


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="multiwoz21")
    parser.add_argument("--model_path", type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = arg_parser()

    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_directory, 'config.json'), 'r') as f:
        cfg = json.load(f)

    cfg['dataset_name'] = args.dataset_name

    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), "info")
    save_config(vars(args), cfg, config_save_path)

    set_seed(args.seed)
    logging.info(f"Seed used: {args.seed}")
    logging.info(f"Batch size: {cfg['batchsz']}")
    logging.info(f"Epochs: {cfg['epoch']}")
    logging.info(f"Learning rate: {cfg['supervised_lr']}")
    logging.info(f"Entropy weight: {cfg['entropy_weight']}")
    logging.info(f"Regularization weight: {cfg['regularization_weight']}")
    logging.info(f"Only use multiwoz like domains: {cfg['multiwoz_like']}")
    logging.info(f"We use: {cfg['data_percentage']*100}% of the data")
    logging.info(f"Dialogue order used: {cfg['dialogue_order']}")

    vector = VectorNodes(dataset_name=args.dataset_name, use_masking=False, filter_state=True)
    manager = PolicyDataVectorizer(dataset_name=args.dataset_name, vector=vector,
                                   percentage=cfg['data_percentage'], dialogue_order=cfg["dialogue_order"])
    policy = EncoderDecoder(**cfg, action_dict=vector.act2vec).to(device=DEVICE)
    try:
        policy.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        logging.info(f"Loaded model from {args.model_path}")
    except:
        logging.info("Didnt load a model")
    agent = MLE_Trainer(manager, cfg, policy)

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

        if e % args.eval_freq == 0:
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
