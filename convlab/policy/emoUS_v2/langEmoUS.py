import json

import torch
import os
from convlab.util.custom_util import model_downloader
from convlab.policy.policy import Policy

from convlab.policy.emoUS_v2.semanticEmoUS import \
    UserActionPolicy as semanticEmoUS
from convlab.policy.emoUS.emoUS import parse_output


class UserActionPolicy(semanticEmoUS):
    def __init__(self, model_checkpoint, mode="language", max_turn=40, **kwargs):
        super().__init__(model_checkpoint, mode, max_turn, **kwargs)
        self.system_utterance = True

    def _generate_action(self, raw_inputs, sys_act: str = None, mode="max", allow_general_intent=True, emotion_mode="normal", emotion=None, golden_action=None):
        self.kg.parse_input(raw_inputs, json.dumps(sys_act))
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

    def predict(self, sys_utt, sys_act, mode="max", allow_general_intent=True, emotion=None):

        allow_general_intent = False
        self.model.eval()
        if not self.add_sys_from_reward:
            self.goal.update_user_goal(action=sys_act, char="sys")
            self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        history = self._get_history()

        input_dict = {"system": sys_utt,
                      "goal": self.goal.get_goal_list(sub_goal_success=self.sub_goal_succ),
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
                    raw_inputs=inputs, sys_act=sys_act, mode=mode, allow_general_intent=allow_general_intent)
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


class UserPolicy(Policy):
    def __init__(self,
                 model_checkpoint="convlab/policy/emoUS/unify/default/EmoUS_default",
                 mode="language",
                 sample=False,
                 action_penalty=False,
                 **kwargs):
        # self.config = config
        self.system_utterance = True
        print("langEmoUS model checkpoint: ", model_checkpoint)
        if sample:
            print("langEmoUS will sample action, but emotion is always max")
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

    def predict(self, sys_utt, sys_act, mode="max"):
        if self.sample:
            mode = "sample"
        else:
            mode = "max"
        response = self.policy.predict(sys_utt, sys_act, mode)
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
