import argparse
import datetime
import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from convlab.dialog_agent.agent import PipelineAgent
from convlab.dialog_agent.env import Environment
from convlab.dialog_agent.session import BiSession
from convlab.dst.rule.multiwoz import RuleDST
from convlab.dst.rule.multiwoz.usr_dst import UserRuleDST
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.policy.tus.multiwoz import util
from convlab.policy.tus.multiwoz.transformer import \
    TransformerActionPrediction
from convlab.policy.tus.multiwoz.TUS import UserPolicy
from convlab.policy.tus.multiwoz.usermanager import TUSDataManager
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_f1(target, result):
    target_len = 0
    result_len = 0
    tp = 0
    for t, r in zip(target, result):
        if t:
            target_len += 1
        if r:
            result_len += 1
        if r == t and t:
            tp += 1
    precision = 0
    recall = 0
    if result_len:
        precision = tp / result_len
    if target_len:
        recall = tp / target_len
    if precision and recall:
        f1_score = 2 / (1 / precision + 1 / recall)
    else:
        f1_score = "NAN"
    return f1_score, precision, recall


def check_device():
    if torch.cuda.is_available():
        print("using GPU")
        return torch.device('cuda')
    else:
        print("using CPU")
        return torch.device('cpu')


def init_logging(log_dir_path, path_suffix=None):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if path_suffix:
        log_file_path = os.path.join(
            log_dir_path, f"{current_time}_{path_suffix}.log")
    else:
        log_file_path = os.path.join(
            log_dir_path, "{}.log".format(current_time))

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    format_str = "%(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    logging.basicConfig(level=logging.DEBUG, handlers=[
                        stderr_handler, file_handler], format=format_str)


class Analysis:
    def __init__(self, config, analysis_dir='user-analysis-result', show_dialog=False, save_dialog=True):
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        self.dialog_dir = os.path.join(analysis_dir, 'dialog')
        if not os.path.exists(self.dialog_dir):
            os.makedirs(self.dialog_dir)
        self.dir = analysis_dir
        self.config = config
        self.device = check_device()
        self.show_dialog = show_dialog
        self.save_dialog = save_dialog
        self.max_turn = 40

    def get_sys(self, sys="rule", load_path=None):
        dst = RuleDST()

        sys = sys.lower()
        if sys == "rule":
            policy = RulePolicy()
        elif sys == "ppo":
            from convlab.policy.ppo import PPO
            if load_path:
                policy = PPO(False, use_action_mask=True, shrink=False)
                policy.load(load_path)
            else:
                policy = PPO.from_pretrained()
        elif sys == "vtrace":
            from convlab.policy.vtrace_rnn_action_embedding import VTRACE_RNN
            policy = VTRACE_RNN(
                is_train=False, seed=0, use_masking=True, shrink=False)
            policy.load(load_path)
        else:
            print(f"Unsupport system type: {sys}")

        return dst, policy

    def get_usr(self, usr="tus", load_path=None):
        # if using "tus", we read config
        # for the other user simulators, we read load_path
        usr = usr.lower()
        if usr == "rule":
            dst_usr = None
            policy_usr = RulePolicy(character='usr')
        elif usr == "tus":
            dst_usr = UserRuleDST()
            policy_usr = UserPolicy(self.config)
        elif usr == "ppo-tus":
            from convlab.policy.ppo.ppo_usr import PPO_USR
            dst_usr = UserRuleDST()
            policy_usr = PPO_USR(pre_trained_config=self.config)
            policy_usr.load(load_path)
        elif usr == "vhus":
            from convlab.policy.vhus.multiwoz import UserPolicyVHUS

            dst_usr = None
            policy_usr = UserPolicyVHUS(
                load_from_zip=True, model_file="vhus_simulator_multiwoz.zip")
        else:
            print(f"Unsupport user type: {usr}")
        # TODO VHUS

        return dst_usr, policy_usr

    def interact_test(self,
                      sys="rule",
                      usr="tus",
                      sys_load_path=None,
                      usr_load_path=None,
                      num_dialog=400,
                      domain=None):
        # TODO need refactor
        seed = 20190827
        torch.manual_seed(seed)
        sys = sys.lower()
        usr = usr.lower()

        sess = self._set_interactive_test(
            sys, usr, sys_load_path, usr_load_path)

        task_success = {
            # 'All_user_sim': [], 'All_evaluator': [], 'total_return': []}
            'complete': [], 'success': [], 'reward': []}

        turn_slot_num = {i: [] for i in range(self.max_turn)}
        turn_domain_num = {i: [] for i in range(self.max_turn)}
        true_max_turn = 0

        for seed in tqdm(range(1000, 1000 + num_dialog)):
            # logging.info(f"Seed: {seed}")

            random.seed(seed)
            np.random.seed(seed)
            sess.init_session()
            # if domain is not none, the user goal must contain certain domain
            if domain:
                domain = domain.lower()
                print(f"check {domain}")
                while 1:
                    if domain in sess.user_agent.policy.get_goal():
                        break
                    sess.user_agent.init_session()
            sys_uttr = []
            actions = 0
            total_return = 0.0
            if self.save_dialog:
                f = open(os.path.join(self.dialog_dir, str(seed)), 'w')
            for turn in range(self.max_turn):
                sys_uttr, usr_uttr, finish, reward = sess.next_turn(sys_uttr)
                if self.show_dialog:
                    print(f"USR: {usr_uttr}")
                    print(f"SYS: {sys_uttr}")
                if self.save_dialog:
                    f.write(f"USR: {usr_uttr}\n")
                    f.write(f"SYS: {sys_uttr}\n")
                actions += len(usr_uttr)
                turn_slot_num[turn].append(len(usr_uttr))
                turn_domain_num[turn].append(self._get_domain_num(usr_uttr))
                total_return += sess.user_agent.policy.policy.get_reward()

                if finish:
                    task_succ = sess.evaluator.task_success()
                    break
            if turn > true_max_turn:
                true_max_turn = turn
            if self.save_dialog:
                f.close()
            # logging.info(f"Return: {total_return}")
            # logging.info(f"Average actions: {actions / (turn+1)}")

            task_success['complete'].append(
                int(sess.user_agent.policy.policy.goal.task_complete()))
            task_success['success'].append(task_succ)
            task_success['reward'].append(total_return)
        task_summary = {key: [0] for key in task_success}
        for key in task_success:
            if task_success[key]:
                task_summary[key][0] = np.average(task_success[key])

        for key in task_success:
            logging.info(
                f'{key} {len(task_success[key])} {task_summary[key][0]}')

        # logging.info("Average action in turn")
        write = {'turn_slot_num': [], 'turn_domain_num': []}
        for turn in turn_slot_num:
            if turn > true_max_turn:
                break
            avg = 0
            if turn_slot_num[turn]:
                avg = sum(turn_slot_num[turn]) / len(turn_slot_num[turn])
            write['turn_slot_num'].append(avg)
            # logging.info(f"turn {turn}: {avg} slots")
        for turn in turn_domain_num:
            if turn > true_max_turn:
                break
            avg = 0
            if turn_domain_num[turn]:
                avg = sum(turn_domain_num[turn]) / len(turn_domain_num[turn])
            write['turn_domain_num'].append(avg)
            # logging.info(f"turn {turn}: {avg} domains")

        # write results
        pd.DataFrame.from_dict(write).to_csv(
            os.path.join(self.dir, f'{sys}-{usr}-turn-statistics.csv'))
        pd.DataFrame.from_dict(task_summary).to_csv(
            os.path.join(self.dir, f'{sys}-{usr}-task-summary.csv'))

    def _get_domain_num(self, action):
        # act: [Intent, Domain, Slot, Value]
        return len(set(act[1] for act in action))

    def _set_interactive_test(self, sys, usr, sys_load_path, usr_load_path):
        dst_sys, policy_sys = self.get_sys(sys, sys_load_path)
        dst_usr, policy_usr = self.get_usr(usr, usr_load_path)

        usr = PipelineAgent(None, dst_usr, policy_usr, None, 'user')
        sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')
        env = Environment(None, usr, None, dst_sys)
        evaluator = MultiWozEvaluator()
        sess = BiSession(sys, usr, None, evaluator)

        return sess

    def data_interact_test(self, test_data, usr="tus", user_mode=None, load_path=None):
        if user_mode:
            # origin_model_name = "-".join(self.config["model_name"].split('-')[:-1])
            self.config["model_name"] = f"model-{user_mode}"

        result = []
        label = []
        dst_usr, policy_usr = self.get_usr(usr=usr, load_path=load_path)

        for dialog_id in test_data:
            if self.show_dialog:
                print(f"dialog_id: {dialog_id}")
            goal = test_data[dialog_id]["goal"]
            sys_act = []
            dst_usr.init_session()
            policy_usr.init_session(goal=goal)
            for turn in range(0, len(test_data[dialog_id]["log"]), 2):
                state = dst_usr.update(sys_act)
                usr_act = policy_usr.predict(state)
                golden_usr = util.parse_dialogue_act(
                    test_data[dialog_id]["log"][turn]["dialog_act"])
                state = dst_usr.update(golden_usr)
                sys_act = util.parse_dialogue_act(
                    test_data[dialog_id]["log"][turn + 1]["dialog_act"])
                result.append(usr_act)
                label.append(golden_usr)
                if self.show_dialog:
                    print(f"---> turn {turn} ")
                    print(f"pre: {usr_act}")
                    print(f"ans: {golden_usr}")
                    print(f"sys: {sys_act}")

        for domain in [None, "attraction", "hotel", "restaurant", "train", "taxi"]:

            statistic = self._data_f1(result, label, domain)
            ana_result = {}
            for stat_type in statistic:
                s = statistic[stat_type]["success"] / \
                    statistic[stat_type]["count"]
                ana_result[stat_type] = s
            ana_result["f1"] = 2/((1/ana_result["precision"])
                                  * (1/ana_result["recall"]))

            print(user_mode)
            for stat_type in ana_result:
                print(f'{stat_type}: {ana_result[stat_type]}')
            col = [c for c in ana_result]
            df_f1 = pd.DataFrame([ana_result[c] for c in col], col)
            if domain:
                df_f1.to_csv(os.path.join(
                    self.dir, f'{domain}-{user_mode}_data_scores.csv'))
            else:
                df_f1.to_csv(os.path.join(
                    self.dir, f'{user_mode}_data_scores.csv'))

    def _extract_domain_related_actions(self, actions, select_domain):
        domain_related_acts = []
        for act in actions:
            domain = act[1].lower()
            if domain == select_domain:
                domain_related_acts.append(act)
        return domain_related_acts

    def _data_f1(self, result, label, domain=None):
        statistic = {}
        for stat_type in ["precision", "recall", "turn_acc"]:
            statistic[stat_type] = {"success": 0, "count": 0}

        for r, l in zip(result, label):
            if domain:
                r = self._extract_domain_related_actions(r, domain)
                l = self._extract_domain_related_actions(l, domain)

            if self._skip(l, r, domain):
                continue
            # check turn accuracy
            turn_acc, tp, fp, fn = self._check(r, l)
            if self.show_dialog:
                print(r, l)
                print(turn_acc, tp, fp, fn)
            if turn_acc:
                statistic["turn_acc"]["success"] += 1
            statistic["turn_acc"]["count"] += 1
            statistic["precision"]["success"] += tp
            statistic["precision"]["count"] += tp + fp
            statistic["recall"]["success"] += tp
            statistic["recall"]["count"] += tp + fn

        return statistic

    @staticmethod
    def _skip(label, result, domain=None):
        ignore = False
        if domain:
            if not label and not result:
                ignore = True
        else:
            if not label:
                ignore = True
            for intent, domain, slot, value in label:
                if intent.lower() in ["thank", "bye"]:
                    ignore = True

        return ignore

    def _check(self, r, l):
        # TODO domain check
        # [['Inform', 'Attraction', 'Addr', 'dontcare']] [['thank', 'general', 'none', 'none']]
        # skip this one
        turn_acc = True
        tp = 0
        fp = 0
        fn = 0
        for a in r:
            is_none_slot, is_in = self._is_in(a, l)
            if is_none_slot:
                continue

            if is_in:
                tp += 1
            else:
                fp += 1
                turn_acc = False

        for a in l:
            is_none_slot, is_in = self._is_in(a, r)
            if is_none_slot:
                continue

            if is_in:
                tp += 1
            else:
                fn += 1
                turn_acc = False

        return turn_acc, tp/2, fp, fn

    @staticmethod
    def _is_in(a, acts):
        is_none_slot = False
        intent, domain, slot, value = a
        if slot.lower() == "none" or domain.lower() == "general":
            is_none_slot = True
            return is_none_slot, True
        if a in acts:
            return is_none_slot, True
        else:
            for i, d, s, v in acts:
                if i == intent and d == domain and s == slot:
                    return is_none_slot, True
            return is_none_slot, False

    def direct_test(self, model, test_data, user_mode=None):

        model = model.to(self.device)
        model.zero_grad()
        model.eval()
        y_lable, y_pred = [], []
        y_turn = []

        result = {}  # old way

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_data, ascii=True, desc="Evaluation"), 0):
                input_feature = data["input"].to(self.device)
                mask = data["mask"].to(self.device)
                label = data["label"].to(self.device)
                output = model(input_feature, mask)
                y_l, y_p, y_t, r = self.parse_result(output, label)
                y_lable += y_l
                y_pred += y_p
                y_turn += y_t
                # old way
                for r_type in r:
                    if r_type not in result:
                        result[r_type] = {"correct": 0, "total": 0}
                    for n in result[r_type]:
                        result[r_type][n] += float(r[r_type][n])
        old_result = {}
        for r_type in result:
            temp = result[r_type]['correct'] / result[r_type]['total']
            old_result[r_type] = [temp]

        pd.DataFrame.from_dict(old_result).to_csv(
            os.path.join(self.dir, f'{user_mode}_old_result.csv'))

        cm = self.model_confusion_matrix(y_lable, y_pred)
        self.summary(y_lable, y_pred, y_turn, cm,
                     file_name=f'{user_mode}_scores.csv')

        return old_result

    def summary(self, y_true, y_pred, y_turn, cm, file_name='scores.csv'):
        f1, pre, rec = get_f1(y_true, y_pred)
        result = {
            'f1': f1,  # metrics.f1_score(y_true, y_pred, average='micro'),
            # metrics.precision_score(y_true, y_pred, average='micro'),
            'precision': pre,
            # metrics.recall_score(y_true, y_pred, average='micro'),
            'recall': rec,
            'none-zero-acc': self.none_zero_acc(cm),
            'turn-acc': sum(y_turn) / len(y_turn)}
        col = [c for c in result]
        df_f1 = pd.DataFrame([result[c] for c in col], col)
        df_f1.to_csv(os.path.join(self.dir, file_name))

    def none_zero_acc(self, cm):
        # ['Unnamed: 0', 'none', '?', 'dontcare', 'sys', 'usr', 'random']
        col = cm.columns[1:]
        num_label = cm.sum(axis=1)
        correct = 0
        for col_name in col:
            correct += cm[col_name][col_name]
        return correct / sum(num_label[1:])

    def model_confusion_matrix(self, y_true, y_pred, file_name='cm.csv', legend=["none", "?", "dontcare", "sys", "usr", "random"]):
        cm = metrics.confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, legend, legend)
        df_cm.to_csv(os.path.join(self.dir, file_name))
        return df_cm

    def parse_result(self, prediction, label):
        _, arg_prediction = torch.max(prediction.data, -1)
        batch_size, token_num = label.shape
        y_true, y_pred = [], []
        y_turn = []
        result = {
            "non-zero": {"correct": 0, "total": 0},
            "total": {"correct": 0, "total": 0},
            "turn": {"correct": 0, "total": 0}
        }

        for batch_num in range(batch_size):

            turn_acc = True  # old way
            turn_success = 1  # new way

            for element in range(token_num):
                result["total"]["total"] += 1
                l = label[batch_num][element].item()
                p = arg_prediction[batch_num][element + 1].item()
                # old way
                if l > 0:
                    result["non-zero"]["total"] += 1
                if p == l:
                    if l > 0:
                        result["non-zero"]["correct"] += 1
                    result["total"]["correct"] += 1
                elif p == 0 and l < 0:
                    result["total"]["correct"] += 1

                else:
                    if l >= 0:
                        turn_acc = False

                # new way
                if l >= 0:
                    y_true.append(l)
                    y_pred.append(p)
                if l >= 0 and l != p:
                    turn_success = 0
            y_turn.append(turn_success)
            # old way
            result["turn"]["total"] += 1
            if turn_acc:
                result["turn"]["correct"] += 1

        return y_true, y_pred, y_turn, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str,
                        default="user-analysis-result")
    parser.add_argument("--user_config", type=str,
                        default="convlab/policy/tus/multiwoz/exp/default.json")
    parser.add_argument("--user_mode", type=str, default="")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--do_direct", action="store_true")
    parser.add_argument("--do_interact", action="store_true")
    parser.add_argument("--do_data", action="store_true")
    parser.add_argument("--show_dialog", action="store_true")
    parser.add_argument("--usr", type=str, default="tus")
    parser.add_argument("--sys", type=str, default="rule")
    parser.add_argument("--num_dialog", type=int, default=400)
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--sys_config", type=str, default="")
    parser.add_argument("--sys_model_dir", type=str,
                        default="convlab/policy/ppo/save/")
    parser.add_argument("--domain", type=str, default="",
                        help="the user goal must contain a specific domain")
    parser.add_argument("--load_path", type=str, default="",
                        help="load path for certain models")

    args = parser.parse_args()

    analysis_dir = os.path.join(args.analysis_dir, f"{args.sys}-{args.usr}")
    if args.version:
        analysis_dir = os.path.join(analysis_dir, args.version)

    if not os.path.exists(os.path.join(analysis_dir)):
        os.makedirs(analysis_dir)

    config = json.load(open(args.user_config))
    init_logging(log_dir_path=os.path.join(analysis_dir, "log"))
    if args.user_mode:
        config["model_name"] = config["model_name"] + '-' + args.user_mode
    # with open(config["all_slot"]) as f:
    #     action_list = [line.strip() for line in f]
    # config["num_token"] = len(action_list)
    if args.use_mask:
        config["domain_mask"] = True

    ana = Analysis(config, analysis_dir=analysis_dir,
                   show_dialog=args.show_dialog)

    if (args.usr == "tus" or args.usr == "ppo-tus") and args.do_data:
        test_data_file = "data/multiwoz/test.json"
        with open(test_data_file) as f:
            test_data = json.load(f)
        if args.user_mode:
            ana.data_interact_test(test_data=test_data,
                                   usr=args.usr,
                                   user_mode=args.user_mode,
                                   load_path=args.load_path)
        else:
            for user_mode in ["loss", "total", "turn", "non-zero"]:
                ana.data_interact_test(test_data=test_data,
                                       usr=args.usr,
                                       user_mode=user_mode,
                                       load_path=args.load_path)

    if args.usr == "tus" and args.do_direct:
        test_data = DataLoader(
            TUSDataManager(
                config, data_dir="data", set_type='test'),
            batch_size=config["batch_size"],
            shuffle=True)

        model = TransformerActionPrediction(config)
        if args.user_mode:
            model.load_state_dict(torch.load(
                os.path.join(config["model_dir"], config["model_name"])))
            print(args.user_mode)
            old_result = ana.direct_test(
                model, test_data, user_mode=args.user_mode)
            print(old_result)
        else:
            for user_mode in ["loss", "total", "turn", "non-zero"]:
                model.load_state_dict(torch.load(
                    os.path.join(config["model_dir"], config["model_name"] + '-' + user_mode)))
                print(user_mode)
                old_result = ana.direct_test(
                    model, test_data, user_mode=user_mode)
                print(old_result)

    if args.do_interact:
        sys_load_path = None
        if args.sys_config:
            _, file_extension = os.path.splitext(args.sys_config)
            # read from config
            if file_extension == ".json":
                sys_config = json.load(open(args.sys_config))
                file_name = f"{sys_config['current_time']}_best_complete_rate_ppo"
                sys_load_path = os.path.join(args.sys_model_dir, file_name)
            # read from file
            else:
                sys_load_path = args.sys_config
        ana.interact_test(sys=args.sys, usr=args.usr,
                          sys_load_path=sys_load_path,
                          num_dialog=args.num_dialog,
                          domain=args.domain)
