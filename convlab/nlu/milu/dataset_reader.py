# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import random
import zipfile
from typing import Dict, List, Any

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MultiLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from convlab.util import load_dataset, load_nlu_data
from overrides import overrides

from convlab.util.file_util import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("milu")
class MILUDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 context_size: int = 0,
                 agent: str = None,
                 use_unified_datasets: bool = False,
                 dataset_name: str = None,
                 random_context_size: bool = True,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._context_size = context_size
        self._agent = agent 
        self.use_unified_datasets = use_unified_datasets
        if self.use_unified_datasets:
            self._dataset_name = dataset_name
            self._dataset = load_dataset(self._dataset_name)
        self._random_context_size = random_context_size
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_delimiter = token_delimiter
        self._sent_tokenizer = PunktSentenceTokenizer()
        self._word_tokenizer = TreebankWordTokenizer()

    @overrides
    def _read(self, file_path):
        if self.use_unified_datasets:
            data_split = file_path
            logger.info("Reading instances from unified dataset %s[%s]", self._dataset_name, data_split)

            data = load_nlu_data(self._dataset, data_split=data_split, speaker=self._agent, use_context=self._context_size>0, context_window_size=self._context_size)[data_split]

            for sample in data:
                utterance = sample['utterance']
                if len(utterance) == 0:
                    continue
                sentences = self._sent_tokenizer.tokenize(utterance)
                sent_spans = self._sent_tokenizer.span_tokenize(utterance)
                tokens = [token for sent in sentences for token in self._word_tokenizer.tokenize(sent)]
                token_spans = [(sent_span[0]+token_span[0], sent_span[0]+token_span[1]) for sent, sent_span in zip(sentences, sent_spans) for token_span in self._word_tokenizer.span_tokenize(sent)]
                tags = ['O'] * len(tokens)

                for da in sample['dialogue_acts']['non-categorical']:
                    if 'start' not in da:
                        # skip da that doesn't have span annotation
                        continue
                    char_start = da['start']
                    char_end = da['end']
                    word_start, word_end = -1, -1
                    for i, token_span in enumerate(token_spans):
                        if char_start == token_span[0]:
                            word_start = i
                        if char_end == token_span[1]:
                            word_end = i + 1
                    if word_start == -1 and word_end == -1:
                        # char span does not match word, maybe there is an error in the annotation, skip
                        print('char span does not match word, skipping')
                        print('\t', 'utteance:', utterance)
                        print('\t', 'value:', utterance[char_start: char_end])
                        print('\t', 'da:', da, '\n')
                        continue
                    intent, domain, slot = da['intent'], da['domain'], da['slot']
                    tags[word_start] = f"B-{intent}+{domain}+{slot}"
                    for i in range(word_start+1, word_end):
                        tags[i] = f"I-{intent}+{domain}+{slot}"

                intents = []
                for da in sample['dialogue_acts']['categorical']:
                    intent, domain, slot, value = da['intent'], da['domain'], da['slot'], da['value'].strip().lower()
                    intent = str((intent, domain, slot, value))
                    intents.append(intent)
                for da in sample['dialogue_acts']['binary']:
                    intent, domain, slot = da['intent'], da['domain'], da['slot']
                    intent = str((intent, domain, slot))
                    intents.append(intent)

                wrapped_tokens = [Token(token) for token in tokens]

                wrapped_context_tokens = []
                num_context = random.randint(0, self._context_size) if self._random_context_size else self._context_size
                if num_context > 0 and len(sample['context']) > 0:
                    for utt in sample['context']:
                        for sent in self._sent_tokenizer.tokenize(utt['utterance']):
                            for token in self._word_tokenizer.tokenize(sent):
                                wrapped_context_tokens.append(Token(token))
                        wrapped_context_tokens.append(Token("SENT_END"))
                else:
                    wrapped_context_tokens = [Token("SENT_END")]

                yield self.text_to_instance(wrapped_context_tokens, wrapped_tokens, tags, intents, sample['dialogue_acts'])
        else:
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            if file_path.endswith("zip"):
                archive = zipfile.ZipFile(file_path, "r")
                data_file = archive.open(os.path.basename(file_path)[:-4])
            else:
                data_file = open(file_path, "r")

            logger.info("Reading instances from lines in file at: %s", file_path)

            dialogs = json.load(data_file)

            for dial_name in dialogs:
                dialog = dialogs[dial_name]["log"]
                context_tokens_list = []
                for i, turn in enumerate(dialog):
                    if self._agent and self._agent == "user" and i % 2 == 1: 
                        context_tokens_list.append(turn["text"].lower().split()+ ["SENT_END"])
                        continue
                    if self._agent and self._agent == "system" and i % 2 == 0: 
                        context_tokens_list.append(turn["text"].lower().split()+ ["SENT_END"])
                        continue

                    tokens = turn["text"].split()

                    dialog_act = {}
                    for dacts in turn["span_info"]:
                        if dacts[0] not in dialog_act:
                            dialog_act[dacts[0]] = []
                        dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4]+1])])

                    spans = turn["span_info"]
                    tags = []
                    for j in range(len(tokens)):
                        for span in spans:
                            if j == span[3]:
                                tags.append("B-"+span[0]+"+"+span[1])
                                break
                            if j > span[3] and j <= span[4]:
                                tags.append("I-"+span[0]+"+"+span[1])
                                break
                        else:
                            tags.append("O")

                    intents = []
                    for dacts in turn["dialog_act"]:
                        for dact in turn["dialog_act"][dacts]:
                            if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                                if dact[1] in ["none", "?", "yes", "no", "dontcare", "do nt care", "do n't care"]:
                                    intents.append(dacts+"+"+dact[0]+"*"+dact[1])

                    for dacts in turn["dialog_act"]:
                        for dact in turn["dialog_act"][dacts]:
                            if dacts not in dialog_act:
                                dialog_act[dacts] = turn["dialog_act"][dacts]
                                break
                            elif dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                                dialog_act[dacts].append(dact)

                    num_context = random.randint(0, self._context_size) if self._random_context_size else self._context_size
                    if len(context_tokens_list) > 0 and num_context > 0:
                        wrapped_context_tokens = [Token(token) for context_tokens in context_tokens_list[-num_context:] for token in context_tokens]
                    else:
                        wrapped_context_tokens = [Token("SENT_END")]
                    wrapped_tokens = [Token(token) for token in tokens]
                    context_tokens_list.append(tokens + ["SENT_END"])

                    yield self.text_to_instance(wrapped_context_tokens, wrapped_tokens, tags, intents, dialog_act)


    def text_to_instance(self, context_tokens: List[Token], tokens: List[Token], tags: List[str] = None,
        intents: List[str] = None, dialog_act: Dict[str, Any] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # print([t.text for t in context_tokens])
        fields["context_tokens"] = TextField(context_tokens, self._token_indexers)
        fields["tokens"] = TextField(tokens, self._token_indexers)
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, fields["tokens"])
        if intents is not None:
            fields["intents"] = MultiLabelField(intents, label_namespace="intent_labels")
        if dialog_act is not None:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
            'dialog_act': dialog_act})
        else:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens], 'dialog_act': {}})
        return Instance(fields)
