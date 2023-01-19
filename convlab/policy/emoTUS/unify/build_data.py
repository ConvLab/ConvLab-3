import json
import os
import sys
from argparse import ArgumentParser

from tqdm import tqdm

from convlab.policy.emoTUS.unify.Goal import Goal, emotion_info
from convlab.policy.genTUS.unify.build_data import \
    DataBuilder as GenTUSDataBuilder
from convlab.policy.genTUS.unify.Goal import transform_data_act
from convlab.policy.tus.unify.util import create_goal, load_experiment_dataset

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="emowoz")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)
    parser.add_argument("--random-order", action="store_true")
    parser.add_argument("--no-status", action="store_true")
    parser.add_argument("--add-history",  action="store_true")
    parser.add_argument("--remove-domain", type=str, default="")

    return parser.parse_args()


class DataBuilder(GenTUSDataBuilder):
    def __init__(self, dataset='emowoz', use_sentiment=False):
        super().__init__(dataset)
        self.use_sentiment = use_sentiment

        self.emotion = {}
        for emotion, index in json.load(open("convlab/policy/emoTUS/emotion.json")).items():
            self.emotion[int(index)] = emotion

        if use_sentiment:
            self.sentiment = {}
            for sentiment, index in json.load(open("convlab/policy/emoTUS/sentiment.json")).items():
                self.sentiment[int(index)] = sentiment
            self.sent2emo = json.load(
                open("convlab/policy/emoTUS/sent2emo.json"))
            # TODO check excited distribution

    def _one_dialog(self, dialog, add_history=True, random_order=False, no_status=False):
        example = []
        history = []

        data_goal = self.norm_domain_goal(create_goal(dialog))
        if not data_goal:
            return example
        user_goal = Goal(goal=data_goal)
        user_info = None
        if self.use_sentiment:
            user_info = emotion_info(dialog)
            # if user_info["user"] == "Impolite":
            #     print(user_info)
            # if "event" in user_info:
            #     print(user_info)

        for turn_id in range(0, len(dialog["turns"]), 2):
            sys_act = self._get_sys_act(dialog, turn_id)

            user_goal.update_user_goal(action=sys_act, char="sys")
            usr_goal_str = self._user_goal_str(
                user_goal, data_goal, random_order, no_status)

            usr_act = self.norm_domain(transform_data_act(
                dialog["turns"][turn_id]["dialogue_acts"]))
            user_goal.update_user_goal(action=usr_act, char="usr")

            # change value "?" to "<?>"
            usr_act = self._modify_act(usr_act)
            usr_emotion = self.emotion[
                dialog["turns"][turn_id]["emotion"][-1]["emotion"]]

            in_str = self._dump_in_str(
                sys_act, usr_goal_str, history, turn_id, add_history, user_info)

            if self.use_sentiment:
                usr_sentiment = self.sentiment[
                    dialog["turns"][turn_id]["emotion"][-1]["sentiment"]]
                out_str = self._dump_out_str(
                    usr_act, dialog["turns"][turn_id]["utterance"], usr_emotion, usr_sentiment)

            else:
                out_str = self._dump_out_str(
                    usr_act, dialog["turns"][turn_id]["utterance"], usr_emotion)

            history.append(usr_act)
            if usr_act:
                example.append({"in": in_str, "out": out_str})

        return example

    def _dump_in_str(self, sys_act, usr_goal_str, history, turn_id, add_history, user_info=None):
        in_str = {}
        in_str["system"] = self._modify_act(sys_act)
        in_str["goal"] = usr_goal_str
        if add_history:
            h = []
            if history:
                h = history[-3:]
            in_str["history"] = h
            in_str["turn"] = str(int(turn_id/2))

        if self.use_sentiment:
            for info in ["event", "user"]:
                if info not in user_info:
                    continue
                in_str[info] = user_info[info]

        return json.dumps(in_str)

    def _dump_out_str(self, usr_act, text, usr_emotion, usr_sentiment=None):
        if self.use_sentiment:
            out_str = {"sentiment": usr_sentiment,
                       "action": usr_act,
                       "emotion": usr_emotion,
                       "text": text}
        else:
            out_str = {"emotion": usr_emotion,
                       "action": usr_act,
                       "text": text}
        return json.dumps(out_str)


"""
TODO
1. add sentiment in the output str
2. check exciting/fearful in user goal (domain)
3. add impolite (user info?)
"""


if __name__ == "__main__":
    args = arg_parser()

    base_name = "convlab/policy/emoTUS/unify/data"
    dir_name = f"{args.dataset}_{args.dial_ids_order}_{args.split2ratio}"
    folder_name = os.path.join(base_name, dir_name)
    remove_domain = args.remove_domain

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    dataset = load_experiment_dataset(
        data_name=args.dataset,
        dial_ids_order=args.dial_ids_order,
        split2ratio=args.split2ratio)
    data_builder = DataBuilder(
        dataset=args.dataset,
        use_sentiment=args.use_sentiment)
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
