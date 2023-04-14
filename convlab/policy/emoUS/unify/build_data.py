import json
import os
import sys
from argparse import ArgumentParser

from tqdm import tqdm

from convlab.policy.emoUS.unify.Goal import Goal, emotion_info
from convlab.policy.genTUS.unify.build_data import \
    DataBuilder as GenTUSDataBuilder
from convlab.policy.genTUS.unify.Goal import transform_data_act
from convlab.policy.tus.unify.util import create_goal, load_experiment_dataset

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="emowoz+dialmage")
    parser.add_argument("--dial-ids-order", type=int, default=0)
    parser.add_argument("--split2ratio", type=float, default=1)
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--add-persona", action="store_true")
    parser.add_argument("--emotion-mid", action="store_true")
    parser.add_argument("--emotion-only", action="store_true")

    return parser.parse_args()


class DataBuilder(GenTUSDataBuilder):
    def __init__(self, dataset='emowoz', **kwargs):
        super().__init__(dataset)
        self.use_sentiment = kwargs.get("use_sentiment", False)
        self.emotion_mid = kwargs.get("emotion_mid", False)
        self.add_persona = kwargs.get("add_persona", False)
        self.emotion_only = kwargs.get("emotion_only", False)

        self.emotion = {}
        for emotion, index in json.load(open("convlab/policy/emoUS/emotion.json")).items():
            self.emotion[int(index)] = emotion

        if use_sentiment:
            self.sentiment = {}
            for sentiment, index in json.load(open("convlab/policy/emoUS/sentiment.json")).items():
                self.sentiment[int(index)] = sentiment
            self.sent2emo = json.load(
                open("convlab/policy/emoUS/sent2emo.json"))
            # TODO check excited distribution

    def _one_dialog(self, dialog, add_history=True, random_order=False, no_status=False):
        example = []
        history = []

        data_goal = self.norm_domain_goal(create_goal(dialog))
        if not data_goal:
            return example
        user_goal = Goal(goal=data_goal)
        user_info = None
        if self.add_persona:
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

        if self.add_persona:
            for info in ["event", "user"]:
                if info not in user_info:
                    continue
                in_str[info] = user_info[info]

        return json.dumps(in_str)

    def _dump_out_str(self, usr_act, text, usr_emotion, usr_sentiment=None):
        if self.use_sentiment and self.emotion_mid:
            out_str = {"sentiment": usr_sentiment,
                       "action": usr_act,
                       "emotion": usr_emotion,
                       "text": text}
        elif self.use_sentiment and not self.emotion_mid:
            out_str = {"sentiment": usr_sentiment,
                       "emotion": usr_emotion,
                       "action": usr_act,
                       "text": text}
        elif not self.use_sentiment and not self.emotion_mid:
            if self.emotion_only:
                out_str = {"emotion": usr_emotion}
            else:
                out_str = {"emotion": usr_emotion,
                           "action": usr_act,
                           "text": text}
        else:
            out_str = {"action": usr_act,
                       "emotion": usr_emotion,
                       "text": text}
        return json.dumps(out_str)


if __name__ == "__main__":
    args = arg_parser()

    base_name = "convlab/policy/emoUS/unify/data"
    dir_name = f"{args.dataset}_{args.dial_ids_order}_{args.split2ratio}"

    use_sentiment = args.use_sentiment
    emotion_mid = args.emotion_mid
    add_persona = args.add_persona

    data_status = [use_sentiment, emotion_mid, add_persona]

    if data_status == [True, True, True]:
        # current sentUS
        dir_name = f"SentUS_{dir_name}"
    elif data_status == [True, True, False]:
        # current sentUS without persona
        dir_name = f"SentUS_noPersona_{dir_name}"
    elif data_status == [False, False, True]:
        # current emoUS with persona
        dir_name = f"EmoUS_{dir_name}"
    elif data_status == [False, False, False]:
        # current emoUS
        dir_name = f"EmoUS_noPersona_{dir_name}"
    elif data_status == [False, True, True]:
        # mid emotion
        dir_name = f"MIDemoUS_{dir_name}"
    elif data_status == [False, True, False]:
        dir_name = f"MIDemoUS_noPersona_{dir_name}"
    elif data_status == [True, False, True]:
        # sentiment followed by emotion, not act
        dir_name = f"SentEmoUS_{dir_name}"
    elif data_status == [True, False, False]:
        # sentiment followed by emotion, not act, without perosna
        dir_name = f"SentEmoUS_noPersona_{dir_name}"
    else:
        print("NOT DEFINED", use_sentiment, add_persona, emotion_mid)

    if args.emotion_only:
        dir_name = dir_name + '_emotion_only'
    print("dir_name", dir_name)

    folder_name = os.path.join(base_name, dir_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    dataset = load_experiment_dataset(
        data_name=args.dataset,
        dial_ids_order=args.dial_ids_order,
        split2ratio=args.split2ratio)
    data_builder = DataBuilder(
        dataset=args.dataset,
        use_sentiment=use_sentiment,
        add_persona=add_persona,
        emotion_mid=emotion_mid,
        emotion_only=args.emotion_only)
    data = data_builder.setup_data(
        raw_data=dataset,
        random_order=False,
        no_status=False,
        add_history=True,
        remove_domain=None)

    for data_type in data:
        file_name = os.path.join(
            folder_name,
            f"{data_type}.json")
        json.dump(data[data_type], open(file_name, 'w'), indent=2)
