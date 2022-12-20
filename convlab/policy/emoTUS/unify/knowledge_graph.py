import json
from random import choices

from convlab.policy.genTUS.token_map import tokenMap
from convlab.policy.genTUS.unify.knowledge_graph import KnowledgeGraph as GenTUSKnowledgeGraph

from transformers import BartTokenizer

DEBUG = False
DATASET = "unify"

# TODO add emotion


class KnowledgeGraph(GenTUSKnowledgeGraph):
    def __init__(self, tokenizer: BartTokenizer, ontology_file=None, dataset="emowoz"):
        super().__init__(tokenizer, ontology_file, dataset="multiwoz")
        data_emotion = json.load(open("convlab/policy/emoTUS/emotion.json"))
        self.emotion = [""]*len(data_emotion)
        for emotion, index in data_emotion.items():
            self.emotion[index] = emotion

        self.kg_map = {"emotion": tokenMap(tokenizer=self.tokenizer)}
        self.prior = {"Neutral": 1,
                      "Disappointed": 1,
                      "Dissatisfied": 1,
                      "Apologetic": 1,
                      "Abusive": 1,
                      "Excited": 1,
                      "Satisfied": 1}

        for emotion in self.emotion:
            self.kg_map["emotion"].add_token(emotion, emotion)

    def get_emotion(self, outputs, mode="max", allow_general_intent=True):
        canidate_list = self.emotion
        score = self._get_max_score(
            outputs, canidate_list, "emotion", weight=self.prior)
        print(score)
        s = self._select(score, mode)

        return score[s]
