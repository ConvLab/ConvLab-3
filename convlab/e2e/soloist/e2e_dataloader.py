import datasets
import jsonlines
import random

# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Corpus for E2E Dialog Modeling"""


import csv

import datasets


_DESCRIPTION = """\
E2E Dialog Modeling
"""

_CITATION = """\
E2E Dialog Modeling
"""

_DOWNLOAD_URL = ""
_WEBPAGE = ""

class UnifiedDialogConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, data_name, **kwargs):
        """BuilderConfig for SuperGLUE.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.2: Fixed non-nondeterminism in ReCoRD.
        # 1.0.1: Change from the pre-release trial version of SuperGLUE (v1.9) to
        #        the full release (v2.0).
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super(UnifiedDialogConfig, self).__init__(version=datasets.Version("1.0.2"), **kwargs)
        self.data_name = data_name
        


class Summarization(datasets.GeneratorBasedBuilder):
    """Summarization"""

    BUILDER_CONFIGS = [
         UnifiedDialogConfig(name='JOINT',data_name='joint'),
         UnifiedDialogConfig(name='TRANSFER',data_name='transfer'),
         UnifiedDialogConfig(name='SINGLE',data_name='single'),
     ]
    
    
    random.seed(2022)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Context": datasets.Value("string"),
                    "Knowledge": datasets.Value("string"),
                    "Response": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                }
            ),
            homepage=_WEBPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):        
        
        data_name = self.config.data_name

        if data_name == 'joint':
            train_path = f'./multiwoz/data/joint_train.jsonl'
            validation_path = f'./multiwoz/data/single_validation.jsonl'
            test_path = f'./multiwoz/data/single_test.jsonl'
        elif data_name == 'transfer':
            train_path = f'./multiwoz/data/transfer_train.jsonl'
            validation_path = f'./multiwoz/data/single_validation.jsonl'
            test_path = f'./multiwoz/data/single_test.jsonl'
        elif data_name == 'single':
            train_path = f'./multiwoz/data/single_train.jsonl'
            validation_path = f'./multiwoz/data/single_validation.jsonl'
            test_path = f'./multiwoz/data/single_test.jsonl'
        else:
            raise('Please specific dataset config.')

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]
    def _generate_examples(self, filepath):
        
        with open(filepath, "r", encoding="utf-8") as reader:
            key = 0
            for item in jsonlines.Reader(reader):
                yield key, item
                key += 1