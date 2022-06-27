# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from pprint import pprint
import torch
from allennlp.common.checks import check_for_gpu
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers import Token

from convlab.util.file_util import cached_path
from convlab.nlu.milu import dataset_reader, model
from convlab.nlu.nlu import NLU
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer

DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "milu_multiwoz_all_context.tar.gz")

class MILU(NLU):
    """Multi-intent language understanding model."""

    def __init__(self,
                archive_file,
                cuda_device,
                model_file,
                context_size):
        """ Constructor for NLU class. """

        self.context_size = context_size
        cuda_device = 0 if torch.cuda.is_available() else DEFAULT_CUDA_DEVICE
        check_for_gpu(cuda_device)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MILU is specified!")

            archive_file = cached_path(model_file)

        archive = load_archive(archive_file,
                            cuda_device=cuda_device)
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TreebankWordTokenizer()

        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.model = archive.model
        self.model.eval()


    def predict(self, utterance, context=list()):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        if len(utterance) == 0:
            return []

        if self.context_size > 0 and len(context) > 0:
            context_tokens = []
            for utt in context[-self.context_size:]:
                for sent in self.sent_tokenizer.tokenize(utt):
                    for token in self.word_tokenizer.tokenize(sent):
                        context_tokens.append(Token(token))
                context_tokens.append(Token("SENT_END"))
        else:
            context_tokens = [Token("SENT_END")]
        sentences = self.sent_tokenizer.tokenize(utterance)
        tokens = [Token(token) for sent in sentences for token in self.word_tokenizer.tokenize(sent)]
        instance = self.dataset_reader.text_to_instance(context_tokens, tokens)
        outputs = self.model.forward_on_instance(instance)

        tuples = []
        for da_type in outputs['dialog_act']:
            for da in outputs['dialog_act'][da_type]:
                tuples.append([da['intent'], da['domain'], da['slot'], da.get('value','')])
        return tuples


if __name__ == "__main__":
    nlu = MILU(archive_file='../output/multiwoz21_user/model.tar.gz', cuda_device=3, model_file=None, context_size=3)
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        pprint(nlu.predict(utt))
