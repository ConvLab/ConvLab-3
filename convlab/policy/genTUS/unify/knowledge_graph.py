import json
from random import choices

from convlab.policy.genTUS.token_map import tokenMap

from transformers import BartTokenizer

DEBUG = False
DATASET = "unify"


class KnowledgeGraph:
    def __init__(self, tokenizer: BartTokenizer, ontology_file=None, dataset="multiwoz21"):
        print("dataset", dataset)
        self.debug = DEBUG
        self.tokenizer = tokenizer

        if "multiwoz" in dataset:
            self.domain_intent = ["inform", "request"]
            self.general_intent = ["thank", "bye"]
        # use sgd dataset intents as default
        else:
            self.domain_intent = ["_inform_intent",
                                  "_negate_intent",
                                  "_affirm_intent",
                                  "inform",
                                  "request",
                                  "affirm",
                                  "negate",
                                  "select",
                                  "_request_alts"]
            self.general_intent = ["thank_you", "goodbye"]

        self.general_domain = "none"
        self.kg_map = {"intent": tokenMap(tokenizer=self.tokenizer)}

        for intent in self.domain_intent + self.general_intent:
            self.kg_map["intent"].add_token(intent, intent)

        self.init()

    def init(self):
        for map_type in ["domain", "slot", "value"]:
            self.kg_map[map_type] = tokenMap(tokenizer=self.tokenizer)
        self.add_token("<?>", "value")

    def parse_input(self, in_str):
        self.init()
        inputs = json.loads(in_str)
        self.sys_act = inputs["system"]
        self.user_goal = {}
        self._add_none_domain()
        for intent, domain, slot, value, _ in inputs["goal"]:
            self._update_user_goal(intent, domain, slot, value, source="goal")

        for intent, domain, slot, value in self.sys_act:
            self._update_user_goal(intent, domain, slot, value, source="sys")

    def _add_none_domain(self):
        self.user_goal["none"] = {"none": "none"}
        # add slot
        self.add_token("none", "domain")
        self.add_token("none", "slot")
        self.add_token("none", "value")

    def _update_user_goal(self, intent, domain, slot, value, source="goal"):

        if value == "?":
            value = "<?>"

        if intent == "request" and source == "sys":
            value = "dontcare"  # user can "dontcare" system request

        if source == "sys" and intent != "request":
            return

        if domain not in self.user_goal:
            self.user_goal[domain] = {}
            self.user_goal[domain]["none"] = ["none"]
            self.add_token(domain, "domain")
            self.add_token("none", "slot")
            self.add_token("none", "value")

        if slot not in self.user_goal[domain]:
            self.user_goal[domain][slot] = []
            self.add_token(domain, "slot")

        if value not in self.user_goal[domain][slot]:
            value = json.dumps(str(value))[1:-1]
            self.user_goal[domain][slot].append(value)
            value = value.replace('"', "'")
            self.add_token(value, "value")

    def add_token(self, token_name, map_type):
        if map_type == "value":
            token_name = token_name.replace('"', "'")
        if not self.kg_map[map_type].token_name_is_in(token_name):
            self.kg_map[map_type].add_token(token_name, token_name)

    def _get_max_score(self, outputs, candidate_list, map_type):
        score = {}
        if not candidate_list:
            print(f"ERROR: empty candidate list for {map_type}")
            score[1] = {"token_id": self._get_token_id(
                "none"), "token_name": "none"}

        for x in candidate_list:
            hash_id = self._get_token_id(x)[0]
            s = outputs[:, hash_id].item()
            score[s] = {"token_id": self._get_token_id(x),
                        "token_name": x}
        return score

    def _select(self, score, mode="max"):
        probs = [s for s in score]
        if mode == "max":
            s = max(probs)
        elif mode == "sample":
            s = choices(probs, weights=probs, k=1)
            s = s[0]

        else:
            print("unknown select mode")

        return s

    def _get_max_domain_token(self, outputs, candidates, map_type, mode="max"):
        score = self._get_max_score(outputs, candidates, map_type)
        s = self._select(score, mode)
        token_id = score[s]["token_id"]
        token_name = score[s]["token_name"]

        return {"token_id": token_id, "token_name": token_name}

    def candidate(self, candidate_type, **kwargs):
        if "intent" in kwargs:
            intent = kwargs["intent"]
        if candidate_type == "intent":
            allow_general_intent = kwargs.get("allow_general_intent", True)
            if allow_general_intent:
                return self.domain_intent + self.general_intent
            else:
                return self.domain_intent
        elif candidate_type == "domain":
            if intent in self.general_intent:
                return [self.general_domain]
            else:
                return [d for d in self.user_goal]
        elif candidate_type == "slot":
            if intent in self.general_intent:
                return ["none"]
            else:
                return self._filter_slot(intent, kwargs["domain"], kwargs["is_mentioned"])
        else:
            if intent in self.general_intent:
                return ["none"]
            elif intent.lower() == "request":
                return ["<?>"]
            else:
                return self._filter_value(intent, kwargs["domain"], kwargs["slot"])

    def get_intent(self, outputs, mode="max", allow_general_intent=True):
        # return intent, token_id_list
        # TODO request?
        canidate_list = self.candidate(
            "intent", allow_general_intent=allow_general_intent)
        score = self._get_max_score(outputs, canidate_list, "intent")
        s = self._select(score, mode)

        return score[s]

    def get_domain(self, outputs, intent, mode="max"):
        if intent in self.general_intent:
            token_name = self.general_domain
            token_id = self.tokenizer(token_name, add_special_tokens=False)
            token_map = {"token_id": token_id['input_ids'],
                         "token_name": token_name}

        elif intent in self.domain_intent:
            # [d for d in self.user_goal]
            domain_list = self.candidate("domain", intent=intent)
            token_map = self._get_max_domain_token(
                outputs=outputs, candidates=domain_list, map_type="domain", mode=mode)
        else:
            if self.debug:
                print("unknown intent", intent)

        return token_map

    def get_slot(self, outputs, intent, domain, mode="max", is_mentioned=False):
        if intent in self.general_intent:
            token_name = "none"
            token_id = self.tokenizer(token_name, add_special_tokens=False)
            token_map = {"token_id": token_id['input_ids'],
                         "token_name": token_name}

        elif intent in self.domain_intent:
            slot_list = self.candidate(
                candidate_type="slot", intent=intent, domain=domain, is_mentioned=is_mentioned)
            token_map = self._get_max_domain_token(
                outputs=outputs, candidates=slot_list, map_type="slot", mode=mode)

        return token_map

    def get_value(self, outputs, intent, domain, slot, mode="max"):
        if intent in self.general_intent or slot.lower() == "none":
            token_name = "none"
            token_id = self.tokenizer(token_name, add_special_tokens=False)
            token_map = {"token_id": token_id['input_ids'],
                         "token_name": token_name}

        elif intent.lower() == "request":
            token_name = "<?>"
            token_id = self.tokenizer(token_name, add_special_tokens=False)
            token_map = {"token_id": token_id['input_ids'],
                         "token_name": token_name}

        elif intent in self.domain_intent:
            # TODO should not none ?
            # value_list = [v for v in self.user_goal[domain][slot]]
            value_list = self.candidate(
                candidate_type="value", intent=intent, domain=domain, slot=slot)

            token_map = self._get_max_domain_token(
                outputs=outputs, candidates=value_list, map_type="value", mode=mode)

        return token_map

    def _filter_slot(self, intent, domain, is_mentioned=True):
        slot_list = []
        for slot in self.user_goal[domain]:
            value_list = self._filter_value(intent, domain, slot)
            if len(value_list) > 0:
                slot_list.append(slot)
        if not is_mentioned and intent.lower() != "request":
            slot_list.append("none")
        return slot_list

    def _filter_value(self, intent, domain, slot):
        value_list = [v for v in self.user_goal[domain][slot]]
        if "none" in value_list:
            value_list.remove("none")
        if intent.lower() != "request":
            if "?" in value_list:
                value_list.remove("?")
            if "<?>" in value_list:
                value_list.remove("<?>")
        # print(f"{intent}-{domain}-{slot}= {value_list}")
        return value_list

    def _get_token_id(self, token):
        return self.tokenizer(token, add_special_tokens=False)["input_ids"]
