import json
import os

import torch
from transformers import BartTokenizer

from convlab.policy.genTUS.ppo.vector import stepGenTUSVector
from convlab.policy.genTUS.stepGenTUSmodel import stepGenTUSmodel
from convlab.policy.genTUS.token_map import tokenMap
from convlab.policy.genTUS.unify.Goal import Goal
from convlab.policy.genTUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.policy import Policy
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.custom_util import model_downloader


DEBUG = False


class UserActionPolicy(Policy):
    def __init__(self, model_checkpoint, mode="semantic", only_action=True, max_turn=40, **kwargs):
        self.mode = mode
        # if mode == "semantic" and only_action:
        #     # only generate semantic action in prediction
        print("model_checkpoint", model_checkpoint)
        self.only_action = only_action
        if self.only_action:
            print("change mode to semantic because only_action=True")
            self.mode = "semantic"
        self.max_in_len = 500
        self.max_out_len = 100 if only_action else 200
        max_act_len = kwargs.get("max_act_len", 2)
        print("max_act_len", max_act_len)
        self.max_action_len = max_act_len
        if "max_act_len" in kwargs:
            self.max_out_len = 30 * self.max_action_len
            print("max_act_len", self.max_out_len)
        self.max_turn = max_turn
        if mode not in ["semantic", "language"]:
            print("Unknown user mode")

        self.reward = {"success":  self.max_turn*2,
                       "fail": self.max_turn*-1}
        self.tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_whole_model = kwargs.get("whole_model", True)
        self.model = stepGenTUSmodel(
            model_checkpoint, train_whole_model=train_whole_model)
        self.model.eval()
        self.model.to(self.device)
        self.model.share_memory()

        self.turn_level_reward = kwargs.get("turn_level_reward", True)
        self.cooperative = kwargs.get("cooperative", True)

        dataset = kwargs.get("dataset", "")
        self.kg = KnowledgeGraph(
            tokenizer=self.tokenizer,
            dataset=dataset)

        self.goal_gen = GoalGenerator()

        self.vector = stepGenTUSVector(
            model_checkpoint, self.max_in_len, self.max_out_len)
        self.norm_reward = False

        self.action_penalty = kwargs.get("action_penalty", False)
        self.usr_act_penalize = kwargs.get("usr_act_penalize", 0)
        self.goal_list_type = kwargs.get("goal_list_type", "normal")
        self.update_mode = kwargs.get("update_mode", "normal")
        self.max_history = kwargs.get("max_history", 3)
        self.init_session()

    def _update_seq(self, sub_seq: list, pos: int):
        for x in sub_seq:
            self.seq[0, pos] = x
            pos += 1

        return pos

    def _generate_action(self, raw_inputs, mode="max", allow_general_intent=True):
        # TODO no duplicate
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq([0], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
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
            # return semantic action. Don't need to generate text
            return self.vector.decode(self.seq[0, :pos])

        # TODO remove illegal action here?

        # get text output
        pos = self._update_seq(self.token_map.get_id("start_text"), pos)

        text = self._get_text(model_input, pos)

        return text

    def generate_text_from_give_semantic(self, raw_inputs, semantic_action):
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq([0], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
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

    def _get_text(self, model_input, pos):
        s_pos = pos
        for i in range(s_pos, self.max_out_len):
            next_token_logits = self.model.get_next_token_logits(
                model_input, self.seq[:1, :pos])
            next_token = torch.argmax(next_token_logits, dim=-1)

            if self._stop_text(next_token):
                # text = self.vector.decode(self.seq[0, s_pos:pos])
                # text = self._norm_str(text)
                # return self.vector.decode(self.seq[0, :s_pos]) + text + '"}'
                break

            pos = self._update_seq([next_token], pos)
        text = self.vector.decode(self.seq[0, s_pos:pos])
        text = self._norm_str(text)
        return self.vector.decode(self.seq[0, :s_pos]) + text + '"}'
        # TODO return None

    def _stop_text(self, next_token):
        if next_token == self.token_map.get_id("end_json")[0]:
            return True
        elif next_token == self.token_map.get_id("end_json_2")[0]:
            return True

        return False

    @staticmethod
    def _norm_str(text: str):
        text = text.strip('"')
        text = text.replace('"', "'")
        text = text.replace('\\', "")
        return text

    def _stop_semantic(self, model_input, pos, act_length=0):

        outputs = self.model.get_next_token_logits(
            model_input, self.seq[:1, :pos])
        tokens = {}
        for token_name in ['sep_act', 'end_act']:
            tokens[token_name] = {
                "token_id": self.token_map.get_id(token_name)}
            hash_id = tokens[token_name]["token_id"][0]
            tokens[token_name]["score"] = outputs[:, hash_id].item()

        if tokens['end_act']["score"] > tokens['sep_act']["score"]:
            terminate = True
        else:
            terminate = False

        if act_length >= self.max_action_len - 1:
            terminate = True

        token_name = "end_act" if terminate else "sep_act"

        return terminate, token_name

    def _get_semantic_action(self, model_input, pos, mode="max", allow_general_intent=True):

        intent = self._get_intent(
            model_input, self.seq[:1, :pos], mode, allow_general_intent)
        pos = self._update_seq(intent["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get domain
        domain = self._get_domain(
            model_input, self.seq[:1, :pos], intent["token_name"], mode)
        pos = self._update_seq(domain["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get slot
        slot = self._get_slot(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], mode)
        pos = self._update_seq(slot["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get value

        value = self._get_value(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], slot["token_name"], mode)
        pos = self._update_seq(value["token_id"], pos)

        return pos

    def _get_intent(self, model_input, generated_so_far, mode="max", allow_general_intent=True):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_intent(next_token_logits, mode, allow_general_intent)

    def _get_domain(self, model_input, generated_so_far, intent, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_domain(next_token_logits, intent, mode)

    def _get_slot(self, model_input, generated_so_far, intent, domain, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        is_mentioned = self.vector.is_mentioned(domain)
        return self.kg.get_slot(next_token_logits, intent, domain, mode, is_mentioned)

    def _get_value(self, model_input, generated_so_far, intent, domain, slot, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_value(next_token_logits, intent, domain, slot, mode)

    def _remove_illegal_action(self, action):
        # Transform illegal action to legal action
        new_action = []
        for act in action:
            if len(act) == 4:
                if "<?>" in act[-1]:
                    act = [act[0], act[1], act[2], "?"]
                if act not in new_action:
                    new_action.append(act)
            else:
                print("illegal action:", action)
        return new_action

    def _parse_output(self, in_str):
        in_str = str(in_str)
        in_str = in_str.replace('<s>', '').replace(
            '<\\s>', '').replace('o"clock', "o'clock")
        action = {"action": [], "text": ""}
        try:
            action = json.loads(in_str)
        except:
            print("invalid action:", in_str)
            print("-"*20)
        return action

    def predict(self, sys_act, mode="max", allow_general_intent=True):
        # raw_sys_act = sys_act
        # sys_act = sys_act[:5]
        # update goal
        # TODO
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
        inputs = json.dumps({"system": sys_act,
                             "goal": self.goal.get_goal_list(),
                             "history": history,
                             "turn": str(int(self.time_step/2))})
        with torch.no_grad():
            raw_output = self._generate_action(
                raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)
        output = self._parse_output(raw_output)
        self.semantic_action = self._remove_illegal_action(output["action"])
        if not self.only_action:
            self.utterance = output["text"]

        # TODO
        if self.is_finish():
            self.semantic_action, self.utterance = self._good_bye()

        # if self.is_finish():
        #     print("terminated")

        # if self.is_finish():
        #     good_bye = self._good_bye()
        #     self.goal.add_usr_da(good_bye)
        #     return good_bye

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        # if self._usr_terminate(usr_action):
        #     print("terminated by user")
        #     self.terminated = True

        del inputs

        if self.mode == "language":
            # print("in", sys_act)
            # print("out", self.utterance)
            return self.utterance
        else:
            return self.semantic_action

    def init_session(self, goal=None):
        self.token_map = tokenMap(tokenizer=self.tokenizer)
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

    def _read_goal(self, data_goal):
        self.goal = Goal(goal=data_goal)

    def _new_goal(self, remove_domain="police", domain_len=None):
        self.goal = Goal(goal_generator=self.goal_gen)
        # keep_generate_goal = True
        # # domain_len = 1
        # while keep_generate_goal:
        #     self.goal = Goal(goal_generator=self.goal_gen,
        #                      goal_list_type=self.goal_list_type,
        #                      update_mode=self.update_mode)
        #     if (domain_len and len(self.goal.domains) != domain_len) or \
        #             (remove_domain and remove_domain in self.goal.domains):
        #         keep_generate_goal = True
        #     else:
        #         keep_generate_goal = False

    def load(self, model_path):
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        # self.model = BartForConditionalGeneration.from_pretrained(
        #     model_checkpoint)

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

    def get_reward(self, sys_response=None):
        self.add_sys_from_reward = False if sys_response is None else True

        if self.add_sys_from_reward:
            self.goal.update_user_goal(action=sys_response, char="sys")
            self.goal.add_sys_da(sys_response)  # for evaluation
            self.sys_acts.append(sys_response)  # for terminate conversation

        if self.is_finish():
            if self.is_success():
                reward = self.reward["success"]
                self.success = True
            else:
                reward = self.reward["fail"]
                self.success = False

        else:
            reward = -1
            if self.turn_level_reward:
                reward += self.turn_reward()

            self.success = None
            # if self.action_penalty:
            #     reward += self._system_action_penalty()

        if self.norm_reward:
            reward = (reward - 20)/60
        return reward

    def _system_action_penalty(self):
        free_action_len = 3
        if len(self.sys_acts) < 1:
            return 0
        # TODO only penalize the slots not in user goal
        # else:
        #     penlaty = 0
        #     for i in range(len(self.sys_acts[-1])):
        #         penlaty += -1*i
        #     return penlaty
        if len(self.sys_acts[-1]) > 3:
            return -1*(len(self.sys_acts[-1])-free_action_len)
        return 0

    def turn_reward(self):
        r = 0
        r += self._new_act_reward()
        r += self._reply_reward()
        r += self._usr_act_len()
        return r

    def _usr_act_len(self):
        last_act = self.usr_acts[-1]
        penalty = 0
        if len(last_act) > 2:
            penalty = (2-len(last_act))*self.usr_act_penalize
        return penalty

    def _new_act_reward(self):
        last_act = self.usr_acts[-1]
        if last_act != self.semantic_action:
            print(f"---> why? last {last_act} usr {self.semantic_action}")
        new_act = []
        for act in last_act:
            if len(self.usr_acts) < 2:
                break
            if act[1].lower() == "general":
                new_act.append(0)
            elif act in self.usr_acts[-2]:
                new_act.append(-1)
            elif act not in self.usr_acts[-2]:
                new_act.append(1)

        return sum(new_act)

    def _reply_reward(self):
        if self.cooperative:
            return self._cooperative_reply_reward()
        else:
            return self._non_cooperative_reply_reward()

    def _non_cooperative_reply_reward(self):
        r = []
        reqts = []
        infos = []
        reply_len = 0
        max_len = 1
        for act in self.sys_acts[-1]:
            if act[0] == "request":
                reqts.append([act[1], act[2]])
        for act in self.usr_acts[-1]:
            if act[0] == "inform":
                infos.append([act[1], act[2]])
        for req in reqts:
            if req in infos:
                if reply_len < max_len:
                    r.append(1)
                elif reply_len == max_len:
                    r.append(0)
                else:
                    r.append(-5)

        if r:
            return sum(r)
        return 0

    def _cooperative_reply_reward(self):
        r = []
        reqts = []
        infos = []
        for act in self.sys_acts[-1]:
            if act[0] == "request":
                reqts.append([act[1], act[2]])
        for act in self.usr_acts[-1]:
            if act[0] == "inform":
                infos.append([act[1], act[2]])
        for req in reqts:
            if req in infos:
                r.append(1)
            else:
                r.append(-1)
        if r:
            return sum(r)
        return 0

    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False

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

    def _good_bye(self):
        if self.is_success():
            return [['thank', 'general', 'none', 'none']], "thank you. bye"
            # if self.mode == "semantic":
            #     return [['thank', 'general', 'none', 'none']]
            # else:
            #     return "bye"
        else:
            return [["bye", "general", "None", "None"]], "bye"
            if self.mode == "semantic":
                return [["bye", "general", "None", "None"]]
            return "bye"

    def _finish_conversation_rule(self):
        if self.is_success():
            return True

        if self.time_step > self.max_turn:
            return True

        if (len(self.sys_acts) > 4) and (self.sys_acts[-1] == self.sys_acts[-2]) and (self.sys_acts[-2] == self.sys_acts[-3]):
            return True
        return False

    def is_terminated(self):
        # Is there any action to say?
        self.is_finish()
        return self.terminated


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
    model_checkpoint = 'convlab/policy/genTUS/unify/experiments/multiwoz21-exp'
    usr_policy = UserPolicy(
        model_checkpoint,
        mode="semantic")
    # usr_policy.policy.load(os.path.join(model_checkpoint, "pytorch_model.bin"))
    usr_nlu = None  # BERTNLU()
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')
    print(usr.policy.get_goal())

    print(usr.response([]))
    print(usr.policy.policy.goal.status)
    print(usr.response([["request", "attraction", "area", "?"]]))
    print(usr.policy.policy.goal.status)
    print(usr.response([["request", "attraction", "area", "?"]]))
    print(usr.policy.policy.goal.status)
