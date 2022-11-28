import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from convlab.policy.rule.multiwoz import RulePolicy
from convlab.policy.tus.unify.Goal import Goal
from convlab.policy.tus.unify.TUS import UserPolicy
from convlab.policy.tus.unify.usermanager import TUSDataManager
from convlab.policy.tus.unify.util import create_goal, parse_dialogue_act
from convlab.util import load_dataset


def check_device():
    if torch.cuda.is_available():
        print("using GPU")
        return torch.device('cuda')
    else:
        print("using CPU")
        return torch.device('cpu')


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

    def get_usr(self, usr="tus", load_path=None):
        # if using "tus", we read config
        # for the other user simulators, we read load_path
        usr = usr.lower()
        if usr == "tus":
            policy_usr = UserPolicy(self.config)
        else:
            print(f"Unsupport user type: {usr}")
        # TODO VHUS

        return policy_usr

    def data_interact_test(self, test_data, usr="tus", user_mode=None, load_path=None):
        if user_mode:
            # origin_model_name = "-".join(self.config["model_name"].split('-')[:-1])
            self.config["model_name"] = f"model-{user_mode}"

        result = []
        label = []
        policy_usr = self.get_usr(usr=usr, load_path=load_path)

        for dialog in tqdm(test_data):
            if self.show_dialog:
                print(f"dialog_id: {dialog['dialog_id']}")
            goal = Goal(create_goal(dialog))

            sys_act = []
            policy_usr.init_session(goal=goal)
            if not policy_usr.get_goal():
                continue
            turn_num = len(dialog["turns"])
            start = 0
            if dialog["turns"][0]["speaker"] == "system":
                start = 1
            for turn_id in range(start, turn_num, 2):
                if turn_id > 0:
                    sys_act = parse_dialogue_act(
                        dialog["turns"][turn_id - 1]["dialogue_acts"])
                usr_act = policy_usr.predict(sys_act)
                golden_usr = parse_dialogue_act(
                    dialog["turns"][turn_id]["dialogue_acts"])
                result.append(usr_act)
                label.append(golden_usr)

        for domain in [None]:

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
            print(df_f1)
            if domain:
                df_f1.to_csv(os.path.join(
                    self.dir, f'{domain}-{user_mode}_data_scores.csv'))
            else:
                df_f1.to_csv(os.path.join(
                    self.config["model_dir"], f'{user_mode}_data_scores.csv'))

    def _extract_domain_related_actions(self, actions, select_domain):
        #
        domain_related_acts = []
        for act in actions:
            domain = act[1].lower()
            if domain == select_domain:
                domain_related_acts.append(act)
        return domain_related_acts

    def _data_f1(self, result, label, domain=None):
        #
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
        #
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
        #
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
        #
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str,
                        default="user-analysis-result")
    parser.add_argument("--user_config", type=str,
                        default="convlab/policy/tus/multiwoz/exp/multiwoz.json")
    parser.add_argument("--user_mode", type=str, default="")
    parser.add_argument("--do_data", action="store_true")
    parser.add_argument("--usr", type=str, default="tus")
    parser.add_argument("--domain", type=str, default="",
                        help="the user goal must contain a specific domain")
    parser.add_argument("--load_path", type=str, default="",
                        help="load path for certain models")
    parser.add_argument("--dataset", type=str, default="multiwoz21",
                        help="data type")
    parser.add_argument("--dial_ids_order", type=int, default=0)

    args = parser.parse_args()

    analysis_dir = os.path.join(f"{args.analysis_dir}-{args.usr}")

    if not os.path.exists(os.path.join(analysis_dir)):
        os.makedirs(analysis_dir)

    config = json.load(open(args.user_config))
    if args.user_mode:
        config["model_name"] = config["model_name"] + '-' + args.user_mode

    # config["model_dir"] = f'{config["model_dir"]}_{args.dial_ids_order}'
    # with open(config["all_slot"]) as f:
    #     action_list = [line.strip() for line in f]
    # config["num_token"] = len(action_list)

    ana = Analysis(config, analysis_dir=analysis_dir)

    if args.usr == "tus" and args.do_data:
        test_data = load_dataset(args.dataset,
                                 dial_ids_order=args.dial_ids_order)["test"]
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
