import json
import os
from argparse import ArgumentParser
from copy import deepcopy

import torch

from convlab.policy.emoUS.emoUS import parse_output
from convlab.policy.emoUS.token_map import tokenMap
from convlab.policy.emoUS.unify.Goal import Goal
from convlab.policy.emoUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.genTUS.stepGenTUS import \
    UserActionPolicy as GenTUSUserActionPolicy
from convlab.policy.policy import Policy
from convlab.util.custom_util import model_downloader

DEBUG = False
# The system will input semantic action to the user policy


class UserActionPolicy(GenTUSUserActionPolicy):
    def __init__(self, model_checkpoint, mode="language", max_turn=40, **kwargs):
        self.use_sentiment = kwargs.get("use_sentiment", False)
        self.add_persona = kwargs.get("add_persona", True)
        self.emotion_mid = kwargs.get("emotion_mid", False)
        self.no_conduct = kwargs.get("no_conduct", False)

        if not os.path.exists(os.path.dirname(model_checkpoint)):
            os.makedirs(os.path.dirname(model_checkpoint))
            model_downloader(os.path.dirname(model_checkpoint),
                             "https://zenodo.org/record/7801525/files/EmoUS_default.zip")

        if mode == "language":
            only_action = False
        elif mode == "semantic":
            only_action = True
        else:
            raise ValueError("mode should be language or semantic")

        super().__init__(model_checkpoint, mode, only_action, max_turn, **kwargs)
        weight = kwargs.get("weight", None)
        self.emotion_weight = {"Neutral": 1,
                               "Fearful": 1,
                               "Dissatisfied": 1,
                               "Apologetic": 1,
                               "Abusive": 1,
                               "Excited": 1,
                               "Satisfied": 1}
        for emotion in self.emotion_weight:
            if emotion in kwargs:
                self.emotion_weight[emotion] = kwargs[emotion]

        self.kg = KnowledgeGraph(
            tokenizer=self.model.tokenizer,
            dataset="emowoz",
            use_sentiment=self.use_sentiment,
            # weight=weight,
            model_type=self.model.model_type,
            **self.emotion_weight)
        data_emotion = json.load(open("convlab/policy/emoUS/emotion.json"))
        self.emotion_list = [""]*len(data_emotion)
        for emotion, index in data_emotion.items():
            self.emotion_list[index] = emotion

        sent2emo = json.load(open("convlab/policy/emoUS/sent2emo.json"))
        self.emo2sent = {}
        for sent, emos in sent2emo.items():
            for emo in emos:
                self.emo2sent[emo] = sent

        self.init_session()

    def estimate_emotion(self, sys_act, mode="max"):
        self.model.eval()
        goal = deepcopy(self.goal)
        goal.update_user_goal(action=sys_act, char="sys")
        history = self._get_history()
        time_step = self.time_step + 2

        input_dict = {"system": sys_act,
                      "goal": goal.get_goal_list(),
                      "history": history,
                      "turn": str(int(time_step/2))}
        if self.add_persona:
            for user, info in self.user_info.items():
                input_dict[user] = info

        inputs = json.dumps(input_dict)
        emotion = self._generate_emotion(inputs, mode)
        return emotion

    def _get_history(self):
        history = []
        if self.usr_acts:
            if self.max_history == 1:
                history = self.usr_acts[-1]
            else:
                history = self.usr_acts[-1*self.max_history:]
        return history

    def predict_emotion_from_text(self, input_dict, mode="max"):
        self.model.eval()
        raw_inputs = json.dumps(input_dict)
        model_input = self.vector.encode(
            raw_inputs, self.max_in_len, do_padding=self.padding)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = 0
        if self.model.model_type == "encoder_decoder":
            pos = self._update_seq([0], 0)
        # else:
        #     pos = self._update_seq([1], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode="normal")
        emotion = self.vector.decode(self.seq[0, :pos]) + '"}'

        return emotion

    def predict(self, sys_act, sys_conduct="neutral", mode="max", allow_general_intent=True, emotion=None):
        if self.no_conduct:
            print("original conduct:", sys_conduct)
            sys_conduct = "neutral"
        allow_general_intent = False
        self.model.eval()
        if not self.add_sys_from_reward:
            self.goal.update_user_goal(action=sys_act, char="sys")
            self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        history = self._get_history()

        input_dict = {"system": sys_act,
                      "conduct": sys_conduct,
                      "goal": self.goal.get_goal_list(),
                      "history": history,
                      "turn": str(int(self.time_step/2))}

        if self.add_persona:
            for user, info in self.user_info.items():
                input_dict[user] = info

        inputs = json.dumps(input_dict)

        with torch.no_grad():
            if emotion is not None:
                raw_output = self.generate_from_emotion(
                    raw_inputs=inputs, emotion=emotion, mode=mode, allow_general_intent=allow_general_intent)
                # print("utt:", output["text"])
            else:
                raw_output = self._generate_action(
                    raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)
        output = parse_output(raw_output)
        self.semantic_action = output["action"]

        if not self.only_action:
            self.utterance = output["text"]

        self.emotion = output["emotion"]
        if self.use_sentiment:
            self.sentiment = output["sentiment"]

        if self.is_finish():
            self.emotion, self.semantic_action, self.utterance = self._good_bye()
            if self.use_sentiment:
                self.sentiment = "Neutral"

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        del inputs
        if self.only_action:
            return self.semantic_action

        return self.utterance

    def _update_sentiment(self, pos, model_input, mode, golden_sentiment=None):
        pos = self._update_seq(
            self.token_map.get_id('start_sentiment'), pos)
        if golden_sentiment:
            sentiment = {"token_name": golden_sentiment}
            pos = self._update_seq(
                self.kg._get_token_id(golden_sentiment), pos)
        else:
            sentiment = self._get_sentiment(
                model_input, self.seq[:1, :pos], mode)
            pos = self._update_seq(sentiment["token_id"], pos)
        return sentiment, pos

    def _update_emotion(self, pos, model_input, mode, emotion_mode, golden_sentiment=None, golden_emotion=None):
        pos = self._update_seq(
            self.token_map.get_id('start_emotion'), pos)
        if golden_emotion:
            pos = self._update_seq(self.kg._get_token_id(golden_emotion), pos)
        else:
            emotion = self._get_emotion(
                model_input, self.seq[:1, :pos], mode, emotion_mode, golden_sentiment)
            pos = self._update_seq(emotion["token_id"], pos)
        return pos

    def _update_semantic_act(self, pos, model_input, mode, allow_general_intent):
        # print("update semantic act")
        mode = "max"
        self.action_prob = []

        for act_len in range(self.max_action_len):
            pos = self._get_semantic_action(
                model_input, pos, mode, allow_general_intent)

            terminate, token_name = self._stop_semantic(
                model_input, pos, act_len)
            pos = self._update_seq(self.token_map.get_id(token_name), pos)

            if terminate:
                break

        return pos

    def _sent_act_emo(self, pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=None, golden_action=None):
        # sent
        golden_sentiment = None
        if golden_emotion:
            golden_sentiment = self.emo2sent[golden_emotion]
        sentiment, pos = self._update_sentiment(
            pos, model_input, mode, golden_sentiment=golden_sentiment)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        # act
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)
        if golden_action:
            pos = self._update_given_act(golden_action, pos)
        else:
            pos = self._update_semantic_act(
                pos, model_input, mode, allow_general_intent)
        # emo
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode, sentiment["token_name"], golden_emotion=golden_emotion)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        return pos

    def _sent_emo_act(self, pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=None, golden_action=None):
        # sent
        golden_sentiment = None
        if golden_emotion:
            golden_sentiment = self.emo2sent[golden_emotion]
        sentiment, pos = self._update_sentiment(
            pos, model_input, mode, golden_sentiment=golden_sentiment)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        # emo
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode, sentiment["token_name"], golden_emotion=golden_emotion)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        # act
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)
        if golden_action:
            pos = self._update_given_act(golden_action, pos)
        else:
            pos = self._update_semantic_act(
                pos, model_input, mode, allow_general_intent)

        return pos

    def _emo_act(self, pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=None, golden_action=None):
        # emo
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode, golden_emotion=golden_emotion)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        # act
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)
        if golden_action:
            pos = self._update_given_act(golden_action, pos)
        else:
            pos = self._update_semantic_act(
                pos, model_input, mode, allow_general_intent)

        return pos

    def _act_emo(self, pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=None, golden_action=None):
        # act
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)
        if golden_action:
            pos = self._update_given_act(golden_action, pos)
        else:
            pos = self._update_semantic_act(
                pos, model_input, mode, allow_general_intent)
        # emo
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode, golden_emotion=golden_emotion)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        return pos

    def _generate_emotion(self, raw_inputs, mode="max", emotion_mode="normal"):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(
            raw_inputs, self.max_in_len, do_padding=self.padding)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = 0
        if self.model.model_type == "encoder_decoder":
            pos = self._update_seq([0], 0)
        # else:
        #     pos = self._update_seq([1], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
        pos = self._update_emotion(
            pos, model_input, mode, emotion_mode)
        emotion = self.vector.decode(self.seq[0, :pos]) + '"}'

        return emotion

    def _generate_action(self, raw_inputs, mode="max", allow_general_intent=True, emotion_mode="normal", emotion=None, golden_action=None):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(
            raw_inputs, self.max_in_len, do_padding=self.padding)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = 0
        if self.model.model_type == "encoder_decoder":
            pos = self._update_seq([0], 0)
        # else:
        #     pos = self._update_seq([1], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)

        if self.use_sentiment and self.emotion_mid:
            pos = self._sent_act_emo(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        elif self.use_sentiment and not self.emotion_mid:
            pos = self._sent_emo_act(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        elif not self.use_sentiment and self.emotion_mid:
            pos = self._act_emo(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        else:  # defalut method
            pos = self._emo_act(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)

        if self.only_action:
            # return semantic action. Don't need to generate text
            return self.vector.decode(self.seq[0, :pos])

        pos = self._update_seq(self.token_map.get_id("start_text"), pos)
        text = self._get_text(model_input, pos)

        return text

    def generate_from_emotion(self, raw_inputs, emotion=None, mode="max", allow_general_intent=True):
        return self._generate_action(raw_inputs, mode, allow_general_intent, emotion_mode="normal", emotion=emotion)

    def generate_text_from_give_semantic(self, raw_inputs, semantic_action, emotion="Neutral"):
        return self._generate_action(raw_inputs, mode="max", allow_general_intent=True, emotion_mode="normal", emotion=emotion, golden_action=semantic_action)

    def _update_given_act(self, semantic_action, pos):
        if len(semantic_action) == 0:
            pos = self._update_seq(self.token_map.get_id("end_act"), pos)

        for act_id, (intent, domain, slot, value) in enumerate(semantic_action):
            pos = self._update_seq(self.kg._get_token_id(intent), pos)
            pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
            pos = self._update_seq(self.kg._get_token_id(domain), pos)
            pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
            pos = self._update_seq(self.kg._get_token_id(slot), pos)
            pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
            pos = self._update_seq(self.kg._get_token_id(value), pos)

            if act_id == len(semantic_action) - 1:
                token_name = "end_act"
            else:
                token_name = "sep_act"
            pos = self._update_seq(self.token_map.get_id(token_name), pos)
        return pos

    def _get_sentiment(self, model_input, generated_so_far, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        return self.kg.get_sentiment(next_token_logits, mode)

    def _get_emotion(self, model_input, generated_so_far, mode="max", emotion_mode="normal", sentiment=None):
        mode = "max"  # emotion is always max
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        return self.kg.get_emotion(next_token_logits, mode, emotion_mode, sentiment)

    def _get_intent(self, model_input, generated_so_far, mode="max", allow_general_intent=True):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_intent(next_token_logits, mode, allow_general_intent)

    def init_session(self, goal=None):
        self.token_map = tokenMap(
            tokenizer=self.model.tokenizer, use_sentiment=self.use_sentiment, model_type=self.model.model_type)
        self.token_map.default(only_action=self.only_action)
        self.time_step = 0
        remove_domain = "police"  # remove police domain in inference

        if not goal:
            self._new_goal(remove_domain=remove_domain)
        else:
            self._read_goal(goal)

        self.vector.init_session(goal=self.goal)

        self.terminated = False
        self.add_sys_from_reward = False
        self.sys_acts = []
        self.usr_acts = []
        self.semantic_action = []
        self.utterance = ""
        self.emotion = "Neutral"
        # TODO sentiment? event? user?
        self.user_info = self.goal.emotion_info()

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
                self.success = True
            else:
                reward = self.reward["fail"]
                self.success = False

        else:
            reward = -1
            if self.use_sentiment:
                if self.sentiment == "Positive":
                    reward += 1
                elif self.sentiment == "Negative":
                    reward -= 1

            self.success = None
        return reward


class UserPolicy(Policy):
    def __init__(self,
                 model_checkpoint="convlab/policy/emoUS/unify/default/EmoUS_default",
                 mode="language",
                 sample=False,
                 action_penalty=False,
                 **kwargs):
        # self.config = config
        print("emoUS model checkpoint: ", model_checkpoint)
        if sample:
            print("EmoUS will sample action, but emotion is always max")
        if not os.path.exists(os.path.dirname(model_checkpoint)):
            os.makedirs(os.path.dirname(model_checkpoint))
            model_downloader(os.path.dirname(model_checkpoint),
                             "https://zenodo.org/record/7801525/files/EmoUS_default.zip")

        self.policy = UserActionPolicy(
            model_checkpoint,
            mode=mode,
            action_penalty=action_penalty,
            **kwargs)
        # self.policy.load(os.path.join(
        #     model_checkpoint, "pytorch_model.bin"))
        self.sample = sample

    def predict(self, sys_act, sys_conduct="neutral", mode="max"):
        if self.sample:
            mode = "sample"
        else:
            mode = "max"
        response = self.policy.predict(sys_act, sys_conduct, mode)
        self.semantic_action = self.policy.semantic_action
        return response

    def estimate_emotion(self, sys_act, mode="max"):
        if self.sample:
            mode = "sample"
        else:
            mode = "max"
        emotion = self.policy.estimate_emotion(sys_act, mode)
        return emotion

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
    parser.add_argument("--model-checkpoint", type=str,
                        default="convlab/policy/emoUS/unify/default/EmoUS_default")
    parser.add_argument("--peft-model-checkpoint", type=str, default="")
    parser.add_argument("--mode", type=str, default="language")
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import time
    from pprint import pprint

    from convlab.dialog_agent import PipelineAgent
    from convlab.util.custom_util import set_seed
    args = arg_parser()
    use_sentiment, emotion_mid = False, False
    set_seed(100)
    # Test semantic level behaviour
    emotion_weight = {"Neutral": 0.9,
                      "Fearful": 1,
                      "Dissatisfied": 1,
                      "Apologetic": 1,
                      "Abusive": 1,
                      "Excited": 1,
                      "Satisfied": 0.9}
    usr_policy = UserPolicy(
        model_checkpoint=args.model_checkpoint,
        mode=args.mode,
        sample=args.sample,
        use_sentiment=use_sentiment,
        emotion_mid=emotion_mid,
        model_type="encoder_decoder",
        peft_model_checkpoint=args.peft_model_checkpoint,
        **emotion_weight)
    # usr_policy.policy.load(os.path.join(model_checkpoint, "pytorch_model.bin"))
    usr_nlu = None  # BERTNLU()
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')
    # usr.init_session()
    usr.init_session()
    start = time.time()

    print(usr.policy.policy.goal)

    print(usr.policy.policy.predict(sys_act=[], emotion="Neutral"))
    print(usr.policy.policy.predict(sys_act=[], emotion="Dissatisfied"))
    print("emotion", usr.policy.policy.emotion)
    print("act", usr.policy.policy.semantic_action)

    end = time.time()
    print("-"*50)
    print("time: ", end - start)
    # print(usr.policy.policy.goal.status)
