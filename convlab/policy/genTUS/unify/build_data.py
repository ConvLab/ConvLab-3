import json
import os
import sys
from argparse import ArgumentParser

from tqdm import tqdm

from convlab.policy.genTUS.unify.Goal import Goal, transform_data_act
from convlab.policy.tus.unify.util import create_goal, load_experiment_dataset


sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="multiwoz21",
                        help="the dataset, such as multiwoz21, sgd, tm1, tm2, and tm3.")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)
    parser.add_argument("--random-order", action="store_true")
    parser.add_argument("--no-status", action="store_true")
    parser.add_argument("--add-history",  action="store_true")
    parser.add_argument("--remove-domain", type=str, default="")

    return parser.parse_args()

class DataBuilder:
    def __init__(self, dataset='multiwoz21'):
        self.dataset = dataset

    def setup_data(self,
                   raw_data,
                   random_order=False,
                   no_status=False,
                   add_history=False,
                   remove_domain=None):
        examples = {data_split: {"dialog": []} for data_split in raw_data}

        for data_split, dialogs in raw_data.items():
            for dialog in tqdm(dialogs, ascii=True):
                example = self._one_dialog(dialog=dialog,
                                           add_history=add_history,
                                           random_order=random_order,
                                           no_status=no_status)
                examples[data_split]["dialog"] += example

        return examples

    def _one_dialog(self, dialog, add_history=True, random_order=False, no_status=False):
        example = []
        history = []

        data_goal = self.norm_domain_goal(create_goal(dialog))
        if not data_goal:
            return example
        user_goal = Goal(goal=data_goal)

        for turn_id in range(0, len(dialog["turns"]), 2):
            sys_act = self._get_sys_act(dialog, turn_id)

            user_goal.update_user_goal(action=sys_act, char="sys")
            usr_goal_str = self._user_goal_str(user_goal, data_goal, random_order, no_status)

            usr_act = self.norm_domain(transform_data_act(
                dialog["turns"][turn_id]["dialogue_acts"]))
            user_goal.update_user_goal(action=usr_act, char="usr")

            # change value "?" to "<?>"
            usr_act = self._modify_act(usr_act)

            in_str = self._dump_in_str(sys_act, usr_goal_str, history, turn_id, add_history)
            out_str = self._dump_out_str(usr_act, dialog["turns"][turn_id]["utterance"])

            history.append(usr_act)
            if usr_act:
                example.append({"in": in_str, "out": out_str})

        return example

    def _get_sys_act(self, dialog, turn_id):
        sys_act = []
        if turn_id > 0:
            sys_act = self.norm_domain(transform_data_act(
                dialog["turns"][turn_id - 1]["dialogue_acts"]))
        return sys_act

    def _user_goal_str(self, user_goal, data_goal, random_order, no_status):
        if random_order:
            usr_goal_str = user_goal.get_goal_list()
        else:
            usr_goal_str = user_goal.get_goal_list(data_goal=data_goal)

        if no_status:
            usr_goal_str = self._remove_status(usr_goal_str)
        return usr_goal_str

    def _dump_in_str(self, sys_act, usr_goal_str, history, turn_id, add_history):
        in_str = {}
        in_str["system"] = self._modify_act(sys_act)
        in_str["goal"] = usr_goal_str
        if add_history:
            h = []
            if history:
                h = history[-3:]
            in_str["history"] = h
            in_str["turn"] = str(int(turn_id/2))

        return json.dumps(in_str)

    def _dump_out_str(self, usr_act, text):
        out_str = {"action": usr_act, "text": text}
        return json.dumps(out_str)

    @staticmethod
    def _norm_intent(intent):
        if intent in ["inform_intent", "negate_intent", "affirm_intent", "request_alts"]:
            return f"_{intent}"
        return intent

    def norm_domain(self, x):
        if not x:
            return x
        norm_result = []
        # print(x)
        for intent, domain, slot, value in x:
            if "_" in domain:
                domain = domain.split('_')[0]
            if not domain:
                domain = "none"
            if not slot:
                slot = "none"
            if not value:
                if intent == "request":
                    value = "<?>"
                else:
                    value = "none"
            norm_result.append([self._norm_intent(intent), domain, slot, value])
        return norm_result

    def norm_domain_goal(self, x):
        if not x:
            return x
        norm_result = []
        # take care of the order!
        for domain, intent, slot, value in x:
            if "_" in domain:
                domain = domain.split('_')[0]
            if not domain:
                domain = "none"
            if not slot:
                slot = "none"
            if not value:
                if intent == "request":
                    value = "<?>"
                else:
                    value = "none"
            norm_result.append([domain, self._norm_intent(intent), slot, value])
        return norm_result

    @staticmethod
    def _remove_status(goal_list):
        new_list = [[goal[0], goal[1], goal[2], goal[3]]
                    for goal in goal_list]
        return new_list

    @staticmethod
    def _modify_act(act):
        new_act = []
        for i, d, s, value in act:
            if value == "?":
                new_act.append([i, d, s, "<?>"])
            else:
                new_act.append([i, d, s, value])
        return new_act


if __name__ == "__main__":
    args = arg_parser()

    base_name = "convlab/policy/genTUS/unify/data"
    dir_name = f"{args.dataset}_{args.dial_ids_order}_{args.split2ratio}"
    folder_name = os.path.join(base_name, dir_name)
    remove_domain = args.remove_domain

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    dataset = load_experiment_dataset(
        data_name=args.dataset,
        dial_ids_order=args.dial_ids_order,
        split2ratio=args.split2ratio)
    data_builder = DataBuilder(dataset=args.dataset)
    data = data_builder.setup_data(
        raw_data=dataset,
        random_order=args.random_order,
        no_status=args.no_status,
        add_history=args.add_history,
        remove_domain=remove_domain)

    for data_type in data:
        if remove_domain:
            file_name = os.path.join(
                folder_name,
                f"no{remove_domain}_{data_type}.json")
        else:
            file_name = os.path.join(
                folder_name,
                f"{data_type}.json")
        json.dump(data[data_type], open(file_name, 'w'), indent=2)
