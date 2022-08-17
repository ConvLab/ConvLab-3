import json

import torch
from convlab.policy.genTUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.genTUS.token_map import tokenMap
from convlab.policy.tus.unify.Goal import Goal
from transformers import BartTokenizer


class stepGenTUSVector:
    def __init__(self, model_checkpoint, max_in_len=400, max_out_len=80, allow_general_intent=True):
        self.tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
        self.vocab = len(self.tokenizer)
        self.max_in_len = max_in_len
        self.max_out_len = max_out_len
        self.token_map = tokenMap(tokenizer=self.tokenizer)
        self.token_map.default(only_action=True)
        self.kg = KnowledgeGraph(self.tokenizer)
        self.mentioned_domain = []
        self.allow_general_intent = allow_general_intent
        self.candidate_num = 5
        if self.allow_general_intent:
            print("---> allow_general_intent")

    def init_session(self, goal: Goal):
        self.goal = goal
        self.mentioned_domain = []

    def encode(self, raw_inputs, max_length, return_tensors="pt", truncation=True):
        model_input = self.tokenizer(raw_inputs,
                                     max_length=max_length,
                                     return_tensors=return_tensors,
                                     truncation=truncation,
                                     padding="max_length")
        return model_input

    def decode(self, generated_so_far, skip_special_tokens=True):
        output = self.tokenizer.decode(
            generated_so_far, skip_special_tokens=skip_special_tokens)
        return output

    def state_vectorize(self, action, history, turn):
        self.goal.update_user_goal(action=action)
        inputs = json.dumps({"system": action,
                             "goal": self.goal.get_goal_list(),
                             "history": history,
                             "turn": str(turn)})
        inputs = self.encode(inputs, self.max_in_len)
        s_vec, action_mask = inputs["input_ids"][0], inputs["attention_mask"][0]

        return s_vec, action_mask

    def action_vectorize(self, action, s=None):
        # action:  [[intent, domain, slot, value], ...]
        vec = {"vector": torch.tensor([]), "mask": torch.tensor([])}
        if s is not None:
            raw_inputs = self.decode(s[0])
            self.kg.parse_input(raw_inputs)

        self._append(vec, self._get_id("<s>"))
        self._append(vec, self.token_map.get_id('start_json'))
        self._append(vec, self.token_map.get_id('start_act'))

        act_len = len(action)
        for i, (intent, domain, slot, value) in enumerate(action):
            if value == '?':
                value = '<?>'
            c_idx = {x: None for x in ["intent", "domain", "slot", "value"]}

            if s is not None:
                c_idx["intent"] = self._candidate_id(self.kg.candidate(
                    "intent", allow_general_intent=self.allow_general_intent))
                c_idx["domain"] = self._candidate_id(self.kg.candidate(
                    "domain", intent=intent))
                c_idx["slot"] = self._candidate_id(self.kg.candidate(
                    "slot", intent=intent, domain=domain, is_mentioned=self.is_mentioned(domain)))
                c_idx["value"] = self._candidate_id(self.kg.candidate(
                    "value", intent=intent, domain=domain, slot=slot))

            self._append(vec, self._get_id(intent), c_idx["intent"])
            self._append(vec, self.token_map.get_id('sep_token'))
            self._append(vec, self._get_id(domain), c_idx["domain"])
            self._append(vec, self.token_map.get_id('sep_token'))
            self._append(vec, self._get_id(slot), c_idx["slot"])
            self._append(vec, self.token_map.get_id('sep_token'))
            self._append(vec, self._get_id(value), c_idx["value"])

            c_idx = [0]*self.candidate_num
            c_idx[0] = self.token_map.get_id('end_act')[0]
            c_idx[1] = self.token_map.get_id('sep_act')[0]
            if i == act_len - 1:
                x = self.token_map.get_id('end_act')
            else:
                x = self.token_map.get_id('sep_act')

            self._append(vec, x, c_idx)

        self._append(vec, self._get_id("</s>"))

        # pad
        if len(vec["vector"]) < self.max_out_len:
            pad_len = self.max_out_len-len(vec["vector"])
            self._append(vec, x=torch.tensor([1]*pad_len))
        for vec_type in vec:
            vec[vec_type] = vec[vec_type].to(torch.int64)

        return vec

    def _append(self, vec, x, candidate=None):
        if type(x) is list:
            x = torch.tensor(x)
        mask = self._mask(x, candidate)
        vec["vector"] = torch.cat((vec["vector"], x), dim=-1)
        vec["mask"] = torch.cat((vec["mask"], mask), dim=0)

    def _mask(self, idx, c_idx=None):
        mask = torch.zeros(len(idx), self.candidate_num)
        mask[:, 0] = idx
        if c_idx is not None and len(c_idx) > 1:
            mask[0, :] = torch.tensor(c_idx)

        return mask

    def _candidate_id(self, candidate):
        if len(candidate) > self.candidate_num:
            print(f"too many candidates. Max = {self.candidate_num}")
        c_idx = [0]*self.candidate_num
        for i, idx in enumerate([self._get_id(c)[0] for c in candidate[:self.candidate_num]]):
            c_idx[i] = idx
        return c_idx

    def _get_id(self, value):
        token_id = self.tokenizer(value, add_special_tokens=False)
        return token_id["input_ids"]

    def action_devectorize(self, action_id):
        return self.decode(action_id)

    def update_mentioned_domain(self, semantic_act):
        for act in semantic_act:
            domain = act[1]
            if domain not in self.mentioned_domain:
                self.mentioned_domain.append(domain)

    def is_mentioned(self, domain):
        if domain in self.mentioned_domain:
            return True
        return False
