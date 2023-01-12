import os
import json

import torch

from convlab.policy.emoTUS.token_map import tokenMap
from convlab.policy.emoTUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.genTUS.stepGenTUS import \
    UserActionPolicy as GenTUSUserActionPolicy
from convlab.policy.policy import Policy
from convlab.util.custom_util import model_downloader

DEBUG = False


class UserActionPolicy(GenTUSUserActionPolicy):
    def __init__(self, model_checkpoint, mode="semantic", only_action=True, max_turn=40, **kwargs):
        self.use_sentiment = kwargs.get("use_sentiment", False)
        super().__init__(model_checkpoint, mode, only_action, max_turn, **kwargs)
        print("sentiment", self.use_sentiment)

        self.kg = KnowledgeGraph(
            tokenizer=self.tokenizer,
            dataset="emowoz",
            use_sentiment=self.use_sentiment)
        data_emotion = json.load(open("convlab/policy/emoTUS/emotion.json"))
        self.emotion_list = [""]*len(data_emotion)
        for emotion, index in data_emotion.items():
            self.emotion_list[index] = emotion

        self.init_session()

    def predict(self, sys_act, mode="max", allow_general_intent=True, emotion=None):
        # TODO emotion
        allow_general_intent = False
        self.model.eval()

        if not self.add_sys_from_reward:
            self.goal.update_user_goal(action=sys_act, char="sys")
            self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        history = []
        if self.usr_acts:
            if self.max_history == 1:
                history = self.usr_acts[-1]
            else:
                history = self.usr_acts[-1*self.max_history:]

        # TODO add user info? impolite? -> check self.use_sentiment
        if self.use_sentiment:
            # TODO how to get event and user politeness?
            inputs = json.dumps({"system": sys_act,
                                 "goal": self.goal.get_goal_list(),
                                 "history": history,
                                 "turn": str(int(self.time_step/2))})
        else:
            inputs = json.dumps({"system": sys_act,
                                 "goal": self.goal.get_goal_list(),
                                 "history": history,
                                 "turn": str(int(self.time_step/2))})
        with torch.no_grad():
            if emotion == "all":
                raw_output = self.generate_from_emotion(
                    raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)
                for emo in raw_output:
                    output = self._parse_output(raw_output[emo])
                    print("emo:", emo)
                    print("act:", output["action"])
                    print("utt:", output["text"])
                raw_output = raw_output["Neutral"]
            elif emotion is not None:
                raw_output = self.generate_from_emotion(
                    raw_inputs=inputs, emotion=emotion, mode=mode, allow_general_intent=allow_general_intent)
                for emo in raw_output:
                    output = self._parse_output(raw_output[emo])
                    print("emo:", emo)
                    print("act:", output["action"])
                    print("utt:", output["text"])
                raw_output = raw_output[emotion]
            else:
                raw_output = self._generate_action(
                    raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)
        output = self._parse_output(raw_output)
        self.emotion = output["emotion"]
        print(self.emotion)
        self.semantic_action = self._remove_illegal_action(output["action"])
        if not self.only_action:
            self.utterance = output["text"]

        if self.is_finish():
            self.emotion, self.semantic_action, self.utterance = self._good_bye()

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        del inputs

        if self.mode == "language":
            # print("in", sys_act)
            # print("out", self.utterance)
            return self.utterance
        else:
            return self.semantic_action

    def _parse_output(self, in_str):
        in_str = str(in_str)
        in_str = in_str.replace('<s>', '').replace(
            '<\\s>', '').replace('o"clock', "o'clock")
        action = {"emotion": "Neutral", "action": [], "text": ""}
        if self.use_sentiment:
            action["sentiment"] = "Neutral"

        try:
            action = json.loads(in_str)
        except:
            print("invalid action:", in_str)
            print("-"*20)
        return action

    def _generate_action(self, raw_inputs, mode="max", allow_general_intent=True, emotion_mode="max"):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq([0], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
        if self.use_sentiment:
            sentiment = self._get_sentiment(
                model_input, self.seq[:1, :pos], mode)
            pos = self._update_seq(sentiment["token_id"], pos)
        else:
            emotion = self._get_emotion(
                model_input, self.seq[:1, :pos], mode, emotion_mode)
            pos = self._update_seq(emotion["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)

        # get semantic actions
        for act_len in range(self.max_action_len):
            pos = self._get_semantic_action(
                model_input, pos, mode, allow_general_intent)

            terminate, token_name = self._stop_semantic(
                model_input, pos, act_len)
            pos = self._update_seq(self.token_map.get_id(token_name), pos)

            if terminate:
                break

        if self.only_action:
            return self.vector.decode(self.seq[0, :pos])

        if self.use_sentiment:
            pos = self._update_seq(self.token_map.get_id('start_emotion'), pos)
            emotion = self._get_emotion(
                model_input, self.seq[:1, :pos], mode, emotion_mode, sentiment)
            pos = self._update_seq(emotion["token_id"], pos)
            pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        pos = self._update_seq(self.token_map.get_id("start_text"), pos)
        text = self._get_text(model_input, pos)

        return text

    def generate_from_emotion(self, raw_inputs,  emotion=None, mode="max", allow_general_intent=True):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        responses = {}
        if emotion:
            print("if emotion")
            emotion_list = [emotion]
        else:
            emotion_list = self.emotion_list
            print(emotion_list)
        for emotion in emotion_list:
            # start token
            print("emotion", emotion)
            self.seq = torch.zeros(1, self.max_out_len,
                                   device=self.device).long()
            pos = self._update_seq([0], 0)
            pos = self._update_seq(self.token_map.get_id('start_json'), pos)

            pos = self._update_seq(self.kg._get_token_id(emotion), pos)
            pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
            pos = self._update_seq(self.token_map.get_id('start_act'), pos)

            # get semantic actions
            for act_len in range(self.max_action_len):
                pos = self._get_semantic_action(
                    model_input, pos, mode, allow_general_intent)

                terminate, token_name = self._stop_semantic(
                    model_input, pos, act_len)
                pos = self._update_seq(self.token_map.get_id(token_name), pos)

                if terminate:
                    break

            if self.only_action:
                return self.vector.decode(self.seq[0, :pos])

            pos = self._update_seq(self.token_map.get_id("start_text"), pos)
            text = self._get_text(model_input, pos)
            responses[emotion] = text

        return responses

    def generate_text_from_give_semantic(self, raw_inputs, semantic_action, emotion="Neutral"):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq([0], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
        pos = self._update_seq(self.kg._get_token_id(emotion), pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)

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
        pos = self._update_seq(self.token_map.get_id("start_text"), pos)

        raw_output = self._get_text(model_input, pos)
        return self._parse_output(raw_output)["text"]

    def _get_sentiment(self, model_input, generated_so_far, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        return self.kg.get_sentiment(next_token_logits, mode)

    def _get_emotion(self, model_input, generated_so_far, mode="max", emotion_mode="normal", sentiment=None):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        return self.kg.get_emotion(next_token_logits, mode, emotion_mode, sentiment)

    def _get_intent(self, model_input, generated_so_far, mode="max", allow_general_intent=True):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_intent(next_token_logits, mode, allow_general_intent)

    def init_session(self, goal=None):
        self.token_map = tokenMap(
            tokenizer=self.tokenizer, use_sentiment=self.use_sentiment)
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

    def _good_bye(self):
        # add emotion
        if self.is_success():
            return "Satisfied", [['thank', 'general', 'none', 'none']], "thank you. bye"
        else:
            return "Dissatisfied", [["bye", "general", "None", "None"]], "bye"


class UserPolicy(Policy):
    def __init__(self,
                 model_checkpoint,
                 mode="semantic",
                 only_action=True,
                 sample=False,
                 action_penalty=False,
                 **kwargs):
        # self.config = config
        if not os.path.exists(os.path.dirname(model_checkpoint)):
            os.mkdir(os.path.dirname(model_checkpoint))
            model_downloader(os.path.dirname(model_checkpoint),
                             "https://zenodo.org/record/7372442/files/multiwoz21-exp.zip")

        self.policy = UserActionPolicy(
            model_checkpoint,
            mode=mode,
            only_action=only_action,
            action_penalty=action_penalty,
            **kwargs)
        self.policy.load(os.path.join(
            model_checkpoint, "pytorch_model.bin"))
        self.sample = sample

    def predict(self, sys_act, mode="max"):
        if self.sample:
            mode = "sample"
        else:
            mode = "max"
        response = self.policy.predict(sys_act, mode)
        return response

    def init_session(self, goal=None):
        self.policy.init_session(goal)

    def is_terminated(self):
        return self.policy.is_terminated()

    def get_reward(self, sys_response=None):
        return self.policy.get_reward(sys_response)

    def get_goal(self):
        if hasattr(self.policy, 'get_goal'):
            return self.policy.get_goal()
        return None


if __name__ == "__main__":
    import os

    from convlab.dialog_agent import PipelineAgent
    # from convlab.nlu.jointBERT.multiwoz import BERTNLU
    from convlab.util.custom_util import set_seed

    set_seed(20220220)
    # Test semantic level behaviour
    model_checkpoint = 'convlab/policy/emoTUS/unify/experiments/emowoz_0_1/22-12-05-11-23'
    usr_policy = UserPolicy(
        model_checkpoint,
        mode="language",
        only_action=False)
    # usr_policy.policy.load(os.path.join(model_checkpoint, "pytorch_model.bin"))
    usr_nlu = None  # BERTNLU()
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')
    print(usr.policy.get_goal())

    print(usr.response([]))
    # print(usr.policy.policy.goal.status)
    print(usr.response([["inform", "restaurant", "area", "centre"],
                        ["request", "restaurant", "food", "?"]]))
    # print(usr.policy.policy.goal.status)
    print(usr.response([["request", "restaurant", "price range", "?"]]))
    # print(usr.policy.policy.goal.status)
