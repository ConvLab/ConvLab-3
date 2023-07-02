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


def parse_args():
    parser = ArgumentParser(description='Build data for the model')
    parser.add_argument('--dataset', type=str, default="emowoz+dialmage",
                        help='create from which dataset')
    parser.add_argument('-T', '--T5-regenerate', action="store_true")
    parser.add_argument('--language-style', type=str, default="all",
                        choices=["multiwoz", "dialmage", "all"])
    return parser.parse_args()


class DataBuilder(GenTUSDataBuilder):
    def __init__(self, dataset='emowoz', **kwargs):
        super().__init__(dataset)

        self.emotion = {}
        for emotion, index in json.load(open("convlab/policy/emoUS/emotion.json")).items():
            self.emotion[int(index)] = emotion
        self.T5_regenerate = kwargs.get("T5_regenerate", False)
        self.language_style = kwargs.get("language_style", "multiwoz")
        if self.T5_regenerate:
            from convlab.base_models.t5.nlg.nlg import T5NLG
            self.nlg = T5NLG(speaker="system", context_window_size=0,
                             model_name_or_path="ConvLab/t5-small-nlg-multiwoz21_sgd_tm1_tm2_tm3")

    def _one_dialog(self, dialog, add_history=True, random_order=False, no_status=False):
        example = []
        history = []

        data_goal = self.norm_domain_goal(create_goal(dialog))
        if not data_goal:
            return example
        user_goal = Goal(goal=data_goal)
        user_info = emotion_info(dialog)
        event = {}
        if "event" in user_info:
            event = user_info["event"]

            # if user_info["user"] == "Impolite":
            #     print(user_info)
            # if "event" in user_info:
            #     print(user_info)

        for turn_id in range(0, len(dialog["turns"]), 2):
            data_id = f"{dialog['dialogue_id']}-{turn_id}"

            sys_act = self._get_sys_act(dialog, turn_id)
            # only regenerate dialmage data
            if self.T5_regenerate and "dialmage" in data_id and sys_act:
                sys_utt = self.nlg.generate(sys_act)
            else:
                sys_utt = self._get_sys_utt(dialog, turn_id)
            history.append({"role": "system", "text": sys_utt})

            user_goal.update_user_goal(action=sys_act, char="sys")
            usr_goal_str = self._user_goal_str(
                user_goal, data_goal, random_order, no_status)

            usr_act = self.norm_domain(transform_data_act(
                dialog["turns"][turn_id]["dialogue_acts"]))
            user_goal.update_user_goal(action=usr_act, char="usr")

            usr_emotion = self.emotion[
                dialog["turns"][turn_id]["emotion"][-1]["emotion"]]
            usr_utt = dialog["turns"][turn_id]["utterance"]

            # emotion
            in_str = get_emotion_prompt(
                history, event, user_info, usr_goal_str)
            out_str = usr_emotion

            example.append({"id": f"{data_id}-emotion",
                           "in": in_str, "out": out_str})

            # action
            # Do not include empty action
            if usr_act:
                in_str = get_action_prompt(
                    history, event, user_info, usr_goal_str, usr_emotion)
                out_str = json.dumps(usr_act)
                example.append({"id": f"{data_id}-action",
                                "in": in_str, "out": out_str})

            # utterance
            add_utt_data = True
            if self.language_style == "dialmage" and "dialmage" not in data_id:
                add_utt_data = False
            elif self.language_style == "multiwoz" and "dialmage" in data_id:
                add_utt_data = False

            if add_utt_data and usr_act:
                in_str = get_utterance_prompt(
                    history, event, user_info, usr_goal_str, usr_emotion, usr_act)
                out_str = usr_utt
                example.append({"id": f"{data_id}-utterance",
                                "in": in_str, "out": out_str})

            history.append({"role": "user", "text": usr_utt})
        print(dialog['dialogue_id'])
        return example

    def _dump_in_str(self, sys_act, usr_goal_str, history, turn_id, add_history, user_info=None):
        pass

    def _dump_out_str(self, usr_act, text, usr_emotion, usr_sentiment=None):
        pass


def basic_prompt(history: list, event: dict, user: dict, goal: list):
    prompt = f"You are talking to a dialogue system.\n"
    event_prompt = ""
    if event:
        event_prompt += f"You are "
        event_prompt += " and ".join(
            [f"{feeling} for {domain}" for domain, feeling in event.items()])
        event_prompt += ".\n"
    persona_prompt = f"You are {user['user']}.\n"
    history_prompt = ""
    if history:
        history_prompt += "The conversation so far:\n"
        for turn in history[-3:]:
            history_prompt += f"{turn['role']}: {turn['text']}\n"
    goal_prompt = "The given goal status is:\n"
    for intent, domain, slot, value, status in goal:
        if value == "<?>":
            value = "?"
        goal_prompt += f"The {intent} {domain}-{slot}: {value} is {status}.\n"

    return prompt + event_prompt + persona_prompt + history_prompt + goal_prompt


def get_emotion_prompt(history: list, event: dict, user: dict, goal: list):
    prompt = basic_prompt(history, event, user, goal)
    emotions = ["Neutral", "Dissatisfied", "Fearful",
                "Abusive", "Apologetic", "Satisfied", "Excited"]

    emotion_prompt = f"There are 7 possible emotions: " + \
        ", ".join(emotions) + ".\n"
    emotion_prompt += "According to the user persona, conversation history, and goal status, the emotion of the user is: "

    return prompt + emotion_prompt


def get_action_prompt(history: list, event: dict, user: dict, goal: list, emotion: str):
    prompt = basic_prompt(history, event, user, goal)
    emotion_prompt = f"You feel {emotion}.\n"
    action_prompt = "According to the user persona, conversation history, goal status, and the emotion, e.g.[[intent, domain, slot, value], ...], your action is:"

    return prompt + emotion_prompt + action_prompt


def get_utterance_prompt(history: list, event: dict, user: dict, goal: list, emotion: str, action: list):
    prompt = basic_prompt(history, event, user, goal)
    emotion_prompt = f"You feel {emotion}.\n"
    action_prompt = "Your action is: " + json.dumps(action) + "\n"
    utterance_prompt = "According to the user persona, conversation history, goal status, the emotion and the user action, your utterance is:"
    return prompt + emotion_prompt + action_prompt + utterance_prompt


def main():
    args = parse_args()
    base_name = "convlab/policy/llmforus/unify/data"
    dir_name = f"{args.dataset}_{args.language_style}"
    if args.T5_regenerate:
        dir_name += "_T5"

    folder_name = os.path.join(base_name, dir_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    dataset = load_experiment_dataset(
        data_name=args.dataset,
        dial_ids_order=0,
        split2ratio=1)
    data_builder = DataBuilder(dataset=args.dataset,
                               language_style=args.language_style,
                               T5_regenerate=args.T5_regenerate)

    data = data_builder.setup_data(
        raw_data=dataset)

    for data_type in data:
        file_name = os.path.join(
            folder_name,
            f"{data_type}.json")
        json.dump(data[data_type], open(file_name, 'w'), indent=2)


if __name__ == "__main__":
    main()
