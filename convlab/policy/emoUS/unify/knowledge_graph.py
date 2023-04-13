import json
from random import choices

from convlab.policy.genTUS.token_map import tokenMap
from convlab.policy.genTUS.unify.knowledge_graph import KnowledgeGraph as GenTUSKnowledgeGraph

from transformers import BartTokenizer

DEBUG = False
DATASET = "unify"

# TODO add emotion


class KnowledgeGraph(GenTUSKnowledgeGraph):
    def __init__(self, tokenizer: BartTokenizer, ontology_file=None, dataset="emowoz", use_sentiment=False, weight=None):
        super().__init__(tokenizer, ontology_file, dataset="multiwoz")
        self.use_sentiment = use_sentiment

        if use_sentiment:
            data_sentiment = json.load(
                open("convlab/policy/emoUS/sentiment.json"))
            self.kg_map = {"sentiment": tokenMap(tokenizer=self.tokenizer)}
            self.sentiment = [""]*len(data_sentiment)
            for sentiment, index in data_sentiment.items():
                self.sentiment[index] = sentiment
            for sentiment in self.sentiment:
                self.kg_map["sentiment"].add_token(sentiment, sentiment)
                self.kg_map[sentiment] = tokenMap(tokenizer=self.tokenizer)
            self.sent2emo = json.load(
                open("convlab/policy/emoUS/sent2emo.json"))
            for sent in self.sent2emo:
                for emo in self.sent2emo[sent]:
                    self.kg_map[sent].add_token(emo, emo)

        else:
            data_emotion = json.load(
                open("convlab/policy/emoUS/emotion.json"))
            self.emotion = [""]*len(data_emotion)
            for emotion, index in data_emotion.items():
                self.emotion[index] = emotion
            self.kg_map = {"emotion": tokenMap(tokenizer=self.tokenizer)}
            for emotion in self.emotion:
                self.kg_map["emotion"].add_token(emotion, emotion)

        self.emotion_weight = {"Neutral": 1,
                               "Fearful": 1,
                               "Dissatisfied": 1,
                               "Apologetic": 1,
                               "Abusive": 1,
                               "Excited": 1,
                               "Satisfied": 1}
        self.sentiment_weight = {"Neutral": 1, "Positive": 1, "Negative": 1}

        if weight:
            if use_sentiment:
                self.sentiment_weight["Neutral"] = weight
            else:
                self.emotion_weight["Neutral"] = weight

    def get_sentiment(self, outputs, mode="max"):
        score = self._get_max_score(
            outputs, self.sentiment, "sentiment", weight=self.sentiment_weight)
        s = self._select(score, mode)
        return score[s]

    def get_emotion(self, outputs, mode="max", emotion_mode="normal", sentiment=None):
        if self.use_sentiment:
            if not sentiment:
                print("You are in 'use_sentiment' mode. Please provide sentiment")
            score = self._get_max_score(
                outputs, self.sent2emo[sentiment], "sentiment")
        else:
            if emotion_mode == "normal":
                score = self._get_max_score(
                    outputs, self.emotion, "emotion", weight=self.emotion_weight)
            elif emotion_mode == "no_neutral":
                score = self._get_max_score(
                    outputs, self.emotion[1:], "emotion", weight=self.emotion_weight)
            else:
                print(f"unknown emotion mode: {emotion_mode}")
        s = self._select(score, mode)

        return score[s]
