# -*- coding: utf-8 -*-
# Copyright 2023 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
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
"""Ontology Encoder Model"""

import random
from copy import deepcopy

import torch
from transformers import RobertaModel, BertModel
import numpy as np
from tqdm import tqdm

PARENT_CLASSES = {'bert': BertModel,
                  'roberta': RobertaModel}


def OntologyEncoder(parent_name: str):
    """
    Return the Ontology Encoder model based on the parent transformer model.

    Args:
        parent_name (str): Name of the parent transformer model

    Returns:
        OntologyEncoder (class): Ontology Encoder model
    """
    parent_class = PARENT_CLASSES.get(parent_name.lower())

    class OntologyEncoder(parent_class):
        """Ontology Encoder model based on parent transformer model"""
        def __init__(self, config, args, tokenizer):
            """
            Initialize Ontology Encoder model.

            Args:
                config (transformers.configuration_utils.PretrainedConfig): Configuration of the transformer model
                args (argparse.Namespace): Arguments
                tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): Tokenizer

            Returns:
                OntologyEncoder (class): Ontology Encoder model
            """
            super().__init__(config)

            # Set random seeds
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.n_gpu > 0:
                torch.cuda.manual_seed_all(args.seed)

            self.args = args
            self.config = config
            self.tokenizer = tokenizer

        def _encode_candidates(self, candidates: list) -> torch.tensor:
            """
            Embed candidates

            Args:
                candidates (list): List of candidate descriptions

            Returns:
                feats (torch.tensor): Embeddings of the candidate descriptions
            """
            # Tokenize candidate descriptions
            feats = [self.tokenizer.encode_plus(val, add_special_tokens=True, max_length=self.args.max_candidate_len,
                                                padding='max_length', truncation='longest_first')
                     for val in candidates]

            # Encode tokenized descriptions
            with torch.no_grad():
                feats = {key: torch.tensor([f[key] for f in feats]).to(self.device) for key in feats[0]}
                embedded_feats = self(**feats)  # [num_candidates, max_candidate_len, hidden_dim]

            # Reduce/pool descriptions embeddings if required
            if self.args.set_similarity:
                feats = embedded_feats.last_hidden_state.detach().cpu() #[num_candidates, max_candidate_len, hidden_dim]
            elif self.args.candidate_pooling == 'cls':
                feats = embedded_feats.pooler_output.detach().cpu()  # [num_candidates, hidden_dim]
            elif self.args.candidate_pooling == "mean":
                feats = embedded_feats.last_hidden_state.detach().cpu()
                feats = feats.sum(1)
                feats = torch.nn.functional.layer_norm(feats, feats.size())
                feats = feats.detach().cpu()  # [num_candidates, hidden_dim]

            return feats

        def get_slot_candidate_embeddings(self):
            """
            Get embeddings for slots and candidates

            Args:
                set_type (str): Subset of the dataset being used (train/validation/test)
                save_to_file (bool): Indication of whether to save information to file

            Returns:
                slots (dict): domain-slot description embeddings, candidate embeddings and requestable flag for each domain-slot
            """
            # Set model to eval mode
            self.eval()

            slots = dict()
            for domain, subset in tqdm(self.tokenizer.ontology.items(), desc='Domains'):
                for slot, slot_info in tqdm(subset.items(), desc='Slots'):
                    # Get description or use "domain-slot"
                    if self.args.use_descriptions:
                        desc = slot_info['description']
                    else:
                        desc = f"{domain}-{slot}"

                    # Encode domain-slot pair description
                    slot_emb = self._encode_candidates([desc])[0]

                    # Obtain possible value set and discard requestable value
                    values = deepcopy(slot_info['possible_values'])
                    is_requestable = False
                    if '?' in values:
                        is_requestable = True
                        values.remove('?')

                    # Encode value candidates
                    if values:
                        feats = self._encode_candidates(values)
                    else:
                        feats = None

                    # Store domain-slot description embeddings, candidate embeddings and requestable flag for each domain-slot
                    slots[f"{domain}-{slot}"] = (slot_emb, feats, is_requestable)

            return slots

    return OntologyEncoder
