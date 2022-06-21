import json
import os
from collections import Counter

import torch
from convlab.dst.rule.multiwoz.usr_dst import UserRuleDST
from convlab.policy.tus.multiwoz import util
from convlab.policy.tus.multiwoz.Da2Goal import SysDa2Goal
from convlab.policy.tus.multiwoz.Goal import Goal
from torch.utils.data import Dataset
from tqdm import tqdm

NOT_MENTIONED = "not mentioned"


def dic2list(da2goal):
    action_list = []
    for domain in da2goal:
        for slot in da2goal[domain]:
            if da2goal[domain][slot] is None:
                continue
            act = f"{domain}-{da2goal[domain][slot]}"
            if act not in action_list:
                action_list.append(act)
    return action_list


class TUSDataManager(Dataset):
    def __init__(self,
                 config,
                 data_dir='data',
                 set_type='train',
                 max_turns=12):

        self.config = config
        self.set_type = set_type
        if set_type == "train":
            data = json.load(
                open(os.path.join(data_dir, 'multiwoz/train_modified.json'), 'r'))
        elif set_type == "val":
            data = json.load(
                open(os.path.join(data_dir, 'multiwoz/val.json'), 'r'))
        elif set_type == "test":
            data = json.load(
                open(os.path.join(data_dir, 'multiwoz/test.json'), 'r'))
        else:
            print("UNKNOWN DATA TYPE")

        self.all_values = json.load(open("data/multiwoz/all_value.json"))

        self.remove_domain = self.config.get("remove_domain", "")

        feat_type = self.config.get("feat_type", "binary")
        if feat_type == "binary":
            self.feature_handler = BinaryFeature(self.config)
        self.features = self.process(data, max_turns)

    def __getitem__(self, index):
        return {label: self.features[label][index] if self.features[label] is not None else None
                for label in self.features}

    def __len__(self):
        return self.features['input'].size(0)

    def resample(self, size=None):
        n_dialogues = self.__len__()
        if not size:
            size = n_dialogues

        dialogues = torch.randint(low=0, high=n_dialogues, size=(size,))
        self.features = {
            label: self.features[label][dialogues] for label in self.features}

    def to(self, device):
        self.device = device
        self.features = {label: self.features[label].to(
            device) for label in self.features}

    def process(self, data, max_turns):

        feature = {"id": [], "input": [],
                   "label": [], "mask": [], "domain": []}
        dst = UserRuleDST()
        for dialog_id in tqdm(data, ascii=True, desc="Processing"):
            user_goal = Goal(goal=data[dialog_id]["goal"])

            # if one domain is removed, we skip all data related to this domain
            # remove police at default
            if "police" in user_goal.domains:
                continue

            # only remove domain for training data
            if self.set_type == "train":
                if (self.remove_domain and self.remove_domain in user_goal.domains):
                    continue

            turn_num = len(data[dialog_id]["log"])
            pre_state = {}
            sys_act = []
            self.feature_handler.initFeatureHandeler(user_goal)
            dst.init_session()
            user_mentioned = util.get_user_history(
                data[dialog_id]["log"], self.all_values)

            for turn_id in range(0, turn_num, 2):
                all_action = None
                if self.config.get("all_action", False):
                    all_action = self.action_list
                action_list = user_goal.action_list(
                    user_history=user_mentioned,
                    sys_act=sys_act,
                    all_values=self.all_values)
                if turn_id > 0:
                    # cur_state = data[dialog_id]["log"][turn_id-1]["metadata"]
                    sys_act = util.parse_dialogue_act(
                        data[dialog_id]["log"][turn_id - 1]["dialog_act"])
                cur_state = dst.update(sys_act)["belief_state"]

                user_goal.update_user_goal(sys_act, cur_state)
                usr_act = util.parse_dialogue_act(
                    data[dialog_id]["log"][turn_id]["dialog_act"])
                pre_usr_act = None
                if turn_id - 2 >= 0:
                    pre_usr_act = util.parse_dialogue_act(
                        data[dialog_id]["log"][turn_id]["dialog_act"])
                input_feature, mask = self.feature_handler.get_feature(
                    action_list, user_goal, cur_state, pre_state, sys_act)  # TODO why
                label = self.feature_handler.generate_label(
                    action_list, user_goal, cur_state, usr_act)
                domain_label = self.feature_handler.domain_label(
                    user_goal, usr_act)
                pre_state = dst.update(usr_act)
                pre_state = pre_state["belief_state"]
                feature["id"].append(dialog_id)
                feature["input"].append(input_feature)
                feature["mask"].append(mask)
                feature["label"].append(label)
                feature["domain"].append(domain_label)

        print("label distribution")
        label_distribution = Counter()
        for label in feature["label"]:
            label_distribution += Counter(label)
        print(label_distribution)
        feature["input"] = torch.tensor(feature["input"], dtype=torch.float)
        feature["label"] = torch.tensor(feature["label"], dtype=torch.long)
        feature["mask"] = torch.tensor(feature["mask"], dtype=torch.bool)
        feature["domain"] = torch.tensor(feature["domain"], dtype=torch.float)
        for feat_type in ["input", "label", "mask", "domain"]:
            print("{}: {}".format(feat_type, feature[feat_type].shape))
        return feature


class Feature:
    def __init__(self, config):
        self.config = config
        self.intents = ["inform", "request", "recommend", "select",
                        "book", "nobook", "offerbook", "offerbooked", "nooffer"]
        self.general_intent = ["reqmore", "bye", "thank", "welcome", "greet"]
        self.default_values = ["none", "?", "dontcare"]
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/all_value.json')
        self.all_values = json.load(open(path))

    def initFeatureHandeler(self, goal):
        self.goal = goal
        self.domain_list = goal.domains
        usr = util.parse_user_goal(goal)
        self.constrains = {}  # slot: fulfill
        self.requirements = {}  # slot: fulfill
        self.pre_usr = []
        self.all_slot = None
        self.user_feat_hist = []
        for slot in usr:
            if usr[slot] != "?":
                self.constrains[slot] = NOT_MENTIONED

    def get_feature(self, all_slot, user_goal, cur_state, pre_state=None, sys_action=None, usr_action=None, state_vectorize=False):
        """ 
        given current dialog information and return the input feature 
        user_goal: Goal()
        cur_state: dict, {domain: "semi": {slot: value}, "book": {slot: value, "booked": []}}("metadata" in the data set)
        sys_action: [[intent, domain, slot, value]]
        """
        feature = []
        usr = util.parse_user_goal(user_goal)

        if sys_action and not state_vectorize:
            self.update_constrain(sys_action)

        cur = util.metadata2state(cur_state)
        pre = {}
        if pre_state != None:
            pre = util.metadata2state(pre_state)
        if not self.pre_usr and not state_vectorize:
            self.pre_usr = [0] * len(all_slot)

        usr_act_feat = self.get_user_action_feat(
            all_slot, user_goal, usr_action)

        for slot in all_slot:
            feat = self.slot_feature(
                slot, usr, cur, pre, sys_action, usr_act_feat)
            feature.append(feat)

        if not state_vectorize:
            self.user_feat_hist.append(feature)

        feature, mask = self.pad_feature(max_memory=self.config["window"])

        return feature, mask

    def slot_feature(self, slot, user_goal, current_state, previous_state, sys_action, usr_action):
        pass

    def pad_feature(self, max_memory=5):
        feature = []
        feat_dim = len(self.user_feat_hist[0][0])
        num_token = len(self.user_feat_hist[0])
        num_feat = len(self.user_feat_hist)
        zero_pad = [0] * feat_dim

        for feat_index in range(num_feat - 1, max(num_feat - 1 - max_memory, -1), -1):
            if feat_index == num_feat - 1:
                special_token = self.slot_feature(
                    "CLS", {}, {}, {}, [], [])
            else:
                special_token = self.slot_feature(
                    "SEP", {}, {}, {}, [], [])
            feature += [special_token]
            feature += self.user_feat_hist[feat_index]

        max_len = max_memory * self.config["num_token"]
        if len(feature) < max_len:
            padding = [[0] * feat_dim] * (max_len - len(feature))
            feature += padding
            mask = [False] * len(feature) + [True] * (max_len - len(feature))
        else:
            mask = [False] * max_len

        return feature[:max_len], mask[:max_len]

    def domain_label(self, user_goal, dialog_act):
        labels = [0] * self.config["out_dim"]
        goal_domains = user_goal.domains
        no_domain = True

        for intent, domain, slot, value in dialog_act:
            domain = domain.lower()
            if domain in goal_domains:
                index = goal_domains.index(domain)
                labels[index + 1] = 1
                no_domain = False
        if no_domain:
            labels[0] = 1
        return labels

    def generate_label(self, action_list, user_goal, cur_state, dialog_act):
        # label = "none", "?", "dontcare", "system", "user", "change"

        labels = [-1] * self.config["num_token"]

        usr = util.parse_user_goal(user_goal)
        cur = util.metadata2state(cur_state)
        for intent, domain, slot, value in dialog_act:
            domain = domain.lower()
            value = value.lower()
            slot = slot.lower()
            name = util.act2slot(intent, domain, slot, value, self.all_values)

            if name not in action_list:
                # print(f"Not handle name {name} in getting label")
                continue
            name_id = action_list.index(name)
            if name_id >= self.config["num_token"]:
                continue
            if value == "?":
                labels[name_id] = 1
            elif value == "dontcare":
                labels[name_id] = 2
            elif name in cur and value == cur[name]:
                labels[name_id] = 3
            elif name in usr and value == usr[name]:
                labels[name_id] = 4
            elif (name in cur or name in usr) and value not in [cur.get(name), usr.get(name)]:
                labels[name_id] = 5

        for name in action_list:
            domain = name.split('-')[0]
            name_id = action_list.index(name)
            if name_id < len(labels):
                if labels[name_id] < 0 and domain in self.domain_list:
                    labels[name_id] = 0

        self.pre_usr = labels

        return labels

    def get_user_action_feat(self, all_slot, user_goal, usr_act):
        pass

    def update_constrain(self, action):
        """ 
        update constrain status by system actions
        action = [[intent, domain, slot, name]]
        """
        for intent, domain, slot, value in action:
            domain = domain.lower()
            slot = slot.lower()
            if domain in self.domain_list:
                slot = SysDa2Goal[domain].get(slot, "none")
                slot_name = f"{domain}-{slot}"
            elif domain == "booking":
                if slot.lower() == "ref":
                    continue
                slot = SysDa2Goal[domain].get(slot, "none")
                domain = util.get_booking_domain(
                    slot, value, self.all_values, self.domain_list)
                if not domain:
                    continue  # work around
                slot_name = f"{domain}-{slot}"

            else:
                continue
            if value != "?":
                self.constrains[slot_name] = value

    @staticmethod
    def concatenate_subvectors(vec_list):
        vec = []
        for sub_vec in vec_list:
            vec += sub_vec
        return vec


class BinaryFeature(Feature):
    def __init__(self, config):
        super().__init__(config)

    def slot_feature(self, slot, user_goal, current_state, previous_state, sys_action, usr_action):
        feat = []
        feat += self._special_token(slot)
        feat += self._value_representation(
            slot, current_state.get(slot, NOT_MENTIONED))
        feat += self._value_representation(
            slot, user_goal.get(slot, NOT_MENTIONED))
        feat += self._is_constrain_request(slot, user_goal)
        feat += self._is_fulfill(slot, user_goal)
        if self.config.get("conflict", True):
            feat += self._conflict_check(user_goal, current_state, slot)
        if self.config.get("domain_feat", False):
            # feat += self.domain_feat(slot)
            if slot in ["CLS", "SEP"]:
                feat += [0] * (self.goal.max_domain_len +
                               self.goal.max_slot_len)
            else:
                domain_feat, slot_feat = self.goal.get_slot_id(slot)
                feat += domain_feat + slot_feat
        feat += self._first_mention_detection(
            previous_state, current_state, slot)
        feat += self._just_mention(slot, sys_action)
        feat += self._action_representation(slot, sys_action)
        # need change from 0 to domain predictor
        if slot in ["CLS", "SEP"]:
            feat += [0] * self.config["out_dim"]
        else:
            feat += usr_action[slot]
        return feat

    def get_user_action_feat(self, all_slot, user_goal, usr_act):
        if usr_act:
            usr_label = self.generate_label(
                all_slot, user_goal, {}, usr_act)
            self.pre_usr = usr_label
        usr_act_feat = {}
        for index, slot in zip(self.pre_usr, all_slot):
            usr_act_feat[slot] = util.int2onehot(index, self.config["out_dim"])
        return usr_act_feat

    def _special_token(self, slot):
        special_token = ["CLS", "SEP"]
        feat = [0] * len(special_token)
        if slot in special_token:
            feat[special_token.index(slot)] = 1
        return feat

    def _is_constrain_request(self, feature_slot, user_goal):
        if feature_slot in ["CLS", "SEP"]:
            return [0, 0]
        # [is_constrain, is_request]
        value = user_goal.get(feature_slot, NOT_MENTIONED)
        if value == "?":
            return [0, 1]
        elif value == NOT_MENTIONED:
            return [0, 0]
        else:
            return [1, 0]

    def _is_fulfill(self, feature_slot, user_goal):
        if feature_slot in ["CLS", "SEP"]:
            return [0]

        if feature_slot in user_goal and user_goal.get(feature_slot) == self.constrains.get(feature_slot):
            return [1]
        return [0]

    def _just_mention(self, feature_slot, sys_action):
        """
        the system action just mentioned this slot
        """
        if feature_slot in ["CLS", "SEP"]:
            return [0]
        if not sys_action:
            return [0]
        sys_action_slot = []
        for intent, domain, slot, value in sys_action:
            domain = domain.lower()
            slot = slot.lower()
            value = value.lower()
            if domain == "booking":
                domain = util.get_booking_domain(
                    slot, value, self.all_values, self.domain_list)
            if domain in sys_action:
                action = f"{domain}-{slot}"
                sys_action_slot.append(action)
        if feature_slot in sys_action_slot:
            return [1]
        return [0]

    def _action_representation(self, feature_slot, action):

        gen_vec = [0] * len(self.general_intent)
        # ["none", "?", other]
        intent2act = {intent: [0] * 3 for intent in self.intents}

        if action is None or feature_slot in ["CLS", "SEP"]:
            return self._concatenate_action_vector(intent2act, gen_vec)
        for intent, domain, slot, value in action:
            domain = domain.lower()
            slot = slot.lower()
            value = value.lower()

            # general
            if domain == "general":
                self._update_general_action(gen_vec, intent)
            else:
                if domain == "booking":
                    domain = util.get_booking_domain(
                        slot, value, self.all_values, self.domain_list)
                self._update_intent2act(
                    feature_slot, intent2act,
                    domain, intent, slot, value)
                # TODO special slots, "choice, ref, none"

        return self._concatenate_action_vector(intent2act, gen_vec)

    def _update_general_action(self, vec, intent):
        if intent in self.general_intent:
            vec[self.general_intent.index(intent)] = 1

    def _update_intent2act(self, feature_slot, intent2act, domain, intent, slot, value):
        feature_domain, feature_slot = feature_slot.split('-')
        intent = intent.lower()
        slot = slot.lower()
        value = value.lower()
        if slot == "none" and feature_domain == domain:  # None slot
            intent2act[intent][2] = 1
        elif feature_domain == domain and slot == feature_slot:
            if value == "none":
                intent2act[intent][0] = 1
            elif value == "?":
                intent2act[intent][1] = 1
            else:
                intent2act[intent][2] = 1

    def _concatenate_action_vector(self, intent2act, general):
        feat = []
        for intent in intent2act:
            feat += intent2act[intent]
        feat += general
        return feat

    def _value_representation(self, slot, value):
        if slot in ["CLS", "SEP"]:
            return [0, 0, 0, 0]
        if value == NOT_MENTIONED:
            return [1, 0, 0, 0]
        else:
            temp_vector = [0] * (len(self.default_values) + 1)
            if value in self.default_values:
                temp_vector[self.default_values.index(value)] = 1
            else:
                temp_vector[-1] = 1

            return temp_vector

    def _conflict_check(self, user_goal, system_state, slot):
        # conflict = [1] else [0]
        if slot in ["CLS", "SEP"]:
            return [0]
        usr = user_goal.get(slot, NOT_MENTIONED)
        sys = system_state.get(slot, NOT_MENTIONED)
        if usr in [NOT_MENTIONED, "none", ""] and sys in [NOT_MENTIONED, "none", ""]:
            return [0]

        if usr != sys or (usr == "?" and sys == "?"):
            # print(f"{slot} has different value: {usr} and {sys}.")
            # conflict = uniform(0.2, 1)
            conflict = 1
            return [conflict]
        return [0]

    def _first_mention_detection(self, pre_state, cur_state, slot):
        if slot in ["CLS", "SEP"]:
            return [0]

        first_mention = [1]
        not_first_mention = [0]
        cur = cur_state.get(slot, NOT_MENTIONED)
        if pre_state is None:
            if cur not in [NOT_MENTIONED, "none"]:
                return first_mention
            else:
                return not_first_mention

        pre = pre_state.get(slot, NOT_MENTIONED)

        if pre in [NOT_MENTIONED, "none"] and cur not in [NOT_MENTIONED, "none"]:
            return first_mention

        return not_first_mention  # hasn't been mentioned
