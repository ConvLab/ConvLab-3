import os
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer

# from convlab.policy.emoUS.token_map import tokenMap
from convlab.policy.emoUS.unify.Goal import Goal
# from convlab.policy.emoUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.llmforus.build_data import (get_action_prompt,
                                                get_emotion_prompt,
                                                get_utterance_prompt)
from convlab.policy.llmforus.evaluation import parse_action
from convlab.policy.llmforus.generate_result import (direct_get_action,
                                                     get_emotion, get_model,
                                                     get_utterance)
from convlab.policy.policy import Policy
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import model_downloader


class UserActionPolicy:
    def __init__(self, model_checkpoint, peft_checkpoint="", mode="language", max_turn=40, **kwargs):
        """
        LLMforUS: A user model for Emotion-Aware Dialogue Generation in Task-Oriented Dialogue Systems based on large langue models
        You should provide the model_checkpoint for the LLM, and the peft_checkpoint for the adaptor.
        """
        # TODO make sure the meaning of max_token
        self.max_token = 100
        # currently, this model must run on language mode
        # if mode == "language":
        #     only_action = False
        # elif mode == "semantic":
        #     only_action = True
        # else:
        #     raise ValueError("mode should be language or semantic")
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = get_model(model_checkpoint, peft_checkpoint)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.goal_gen = GoalGenerator()
        self.reward = {"success": 2*max_turn, "fail": -1*max_turn}
        self.max_turn = max_turn

        self.init_session()

    def init_session(self, goal=None):
        self._init_dialog_status()
        self._init_user_goal(goal)
        self._init_internal_records()
        self._default_user_behaviour()

    def _default_user_behaviour(self):
        self.emotion = "Neutral"
        self.semantic_action = []
        self.utterance = ""

    def predict(self, sys_utt, sys_act=None):
        # TODO can we update the goal from utternace?
        if sys_act:
            self.goal.update_user_goal(action=sys_act, char="sys")
            self.sys_acts.append(sys_act)
        else:
            # TODO add nlu module here, sys_utt -> sys_act
            pass
        self._update_history(role="system", text=sys_utt)
        history = self.history[-3:]
        goal = self.goal.get_goal_list()
        self.emotion = self.generate_emotion(
            history=history, goal=goal)
        self.semantic_action = self.generate_action(
            history=history, goal=goal, emotion=self.emotion)
        self.utterance = self.generate_utterance(
            history=history, goal=goal, emotion=self.emotion, action=self.semantic_action)
        self._update_history(role="user", text=self.utterance)
        self.goal.update_user_goal(action=self.semantic_action, char="usr")

        self.time_step += 2
        if self.is_finish():
            # TODO check the stop reason
            self.emotion, self.semantic_action, self.utterance = self._good_bye()

        return self.utterance

    def generate_emotion(self, history, goal):
        input_str = get_emotion_prompt(history=history,
                                       event=self._get_user_event(),
                                       user=self.user_info,
                                       goal=goal)
        emotion = get_emotion(text=input_str,
                              model=self.model.model,
                              tokenizer=self.tokenizer,
                              device=self.device,
                              max_token=self.max_token)

        return emotion

    def generate_action(self, history, goal, emotion):
        input_str = get_action_prompt(history=history,
                                      event=self._get_user_event(),
                                      user=self.user_info,
                                      goal=goal,
                                      emotion=emotion)
        action = direct_get_action(text=input_str,
                                   model=self.model.model,
                                   tokenizer=self.tokenizer,
                                   device=self.device,
                                   max_token=self.max_token)

        return parse_action(action)

    def generate_utterance(self, history, goal, emotion, action):
        input_str = get_utterance_prompt(history=history,
                                         event=self._get_user_event(),
                                         user=self.user_info,
                                         goal=goal,
                                         emotion=emotion,
                                         action=action)
        utterance = get_utterance(text=input_str,
                                  model=self.model.model,
                                  tokenizer=self.tokenizer,
                                  device=self.device,
                                  max_token=self.max_token)
        return utterance

    def _get_user_event(self):
        if "event" in self.user_info:
            return self.user_info["event"]
        return None

    def _update_history(self, role, text):
        # The history includes all dialogue history
        self.history.append({"role": role, "text": text})
        return self.history

    def _init_dialog_status(self):
        self.time_step = 0
        self.terminated = False

    def _init_user_goal(self, goal):
        remove_domain = "police"  # remove police domain in inference
        if not goal:
            self._new_goal(remove_domain=remove_domain)
        else:
            self._read_goal(goal)
        self.user_info = self.goal.emotion_info()

    def _init_internal_records(self):
        self.history = []
        self.sys_acts = []
        self.usr_acts = []

    def _read_goal(self, data_goal):
        self.goal = Goal(goal=data_goal)

    def _new_goal(self, remove_domain="police", domain_len=None):
        self.goal = Goal(goal_generator=self.goal_gen)

    def _good_bye(self):
        # add emotion
        if self.is_success():
            return "Satisfied", [['thank', 'general', 'none', 'none']], "thank you. bye"
        else:
            return "Dissatisfied", [["bye", "general", "None", "None"]], "bye"

    def get_reward(self):
        if self.is_finish():
            if self.is_success():
                reward = self.reward["success"]
            else:
                reward = self.reward["fail"]

        else:
            reward = -1
            # TODO
            # emotional reward?
        return reward

    def is_finish(self):
        # stop by model generation?
        if self._finish_conversation_rule():
            self.terminated = True
            return True
        elif self._usr_terminate():
            self.terminated = True
            return True
        self.terminated = False
        return False

    def is_success(self):
        task_complete = self.goal.task_complete()
        # goal_status = self.goal.all_mentioned()
        # should mentioned all slots
        if task_complete:  # and goal_status["complete"] > 0.6:
            return True
        return False

    def _finish_conversation_rule(self):
        if self.is_success():
            return True

        if self.time_step > self.max_turn:
            return True

        if self.is_loop():
            return True
        return False

    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False

    def is_loop(self):
        # check from system semantic acts
        if self.sys_acts:
            if (len(self.sys_acts) > 4) and (self.sys_acts[-1] == self.sys_acts[-2]) and (self.sys_acts[-2] == self.sys_acts[-3]):
                return True
            return False
        history = []
        for h in reversed(self.history):
            if h["role"] == "system":
                history.append(h["text"])
            if len(history) > 2:
                break
        if len(history) > 2:
            if history[0] == history[1] and history[1] == history[2]:
                return True
        return False

    def is_terminated(self):
        # Is there any action to say?
        self.is_finish()
        return self.terminated

    def get_goal(self):
        if self.goal.raw_goal is not None:
            return self.goal.raw_goal
        goal = {}
        for domain in self.goal.domain_goals:
            if domain not in goal:
                goal[domain] = {}
            for intent in self.goal.domain_goals[domain]:
                if intent == "inform":
                    slot_type = "info"
                elif intent == "request":
                    slot_type = "reqt"
                elif intent == "book":
                    slot_type = "book"
                else:
                    print("unknown slot type")
                if slot_type not in goal[domain]:
                    goal[domain][slot_type] = {}
                for slot, value in self.goal.domain_goals[domain][intent].items():
                    goal[domain][slot_type][slot] = value
        return goal


class UserPolicy(Policy):
    def __init__(self,
                 model_checkpoint="",
                 peft_checkpoint="",
                 mode="language",
                 max_turn=40,
                 **kwargs):
        # self.config = config
        print("LLM model checkpoint:", model_checkpoint)
        print("Adaptor model checkpoint:", peft_checkpoint)
        # if sample:
        #     print("EmoUS will sample action, but emotion is always max")
        # if not os.path.exists(os.path.dirname(model_checkpoint)):
        #     os.makedirs(os.path.dirname(model_checkpoint))
        #     model_downloader(os.path.dirname(model_checkpoint),
        #                      "https://zenodo.org/record/7801525/files/EmoUS_default.zip")

        self.policy = UserActionPolicy(
            model_checkpoint=model_checkpoint,
            peft_checkpoint=peft_checkpoint,
            mode=mode,
            max_turn=max_turn,
            **kwargs)
        # self.policy.load(os.path.join(
        #     model_checkpoint, "pytorch_model.bin"))
        # self.sample = sample

    def predict(self, sys_utt, sys_act=None, mode="max"):
        # if self.sample:
        #     mode = "sample"
        # else:
        #     mode = "max"
        response = self.policy.predict(sys_utt=sys_utt, sys_act=sys_act)
        self.semantic_action = self.policy.semantic_action
        return response

    # def estimate_emotion(self, sys_act, mode="max"):
    #     if self.sample:
    #         mode = "sample"
    #     else:
    #         mode = "max"
    #     emotion = self.policy.estimate_emotion(sys_act, mode)
    #     return emotion

    def init_session(self, goal=None):
        self.policy.init_session(goal)
        self.semantic_action = []

    def is_terminated(self):
        return self.policy.is_terminated()

    def get_reward(self):
        return self.policy.get_reward()

    def get_goal(self):
        if hasattr(self.policy, 'get_goal'):
            return self.policy.get_goal()
        return None

    def get_emotion(self):
        return self.policy.emotion


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--peft-checkpoint", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    from convlab.dialog_agent import PipelineAgent
    from convlab.util.custom_util import set_seed

    set_seed(20220220)

    args = arg_parser()
    usr_policy = UserPolicy(model_checkpoint=args.model_checkpoint,
                            peft_checkpoint=args.peft_checkpoint)
    usr = PipelineAgent(None, None, usr_policy, None, name='user')
    print(usr.policy.get_goal())

    print(usr.response("Hi, what can I help you?"))
    print("emotion", usr.policy.get_emotion())
    print("action", usr.policy.semantic_action)
