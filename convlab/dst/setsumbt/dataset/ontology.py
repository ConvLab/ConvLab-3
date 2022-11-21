# -*- coding: utf-8 -*-
# Copyright 2022 DSML Group, Heinrich Heine University, Düsseldorf
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
"""Create Ontology Embeddings"""

import json
import os
import random
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm


def set_seed(args):
    """
    Set random seeds

    Args:
        args (Arguments class): Arguments class containing seed and number of gpus to use
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def encode_candidates(candidates: list, args, tokenizer, embedding_model) -> torch.tensor:
    """
    Embed candidates

    Args:
        candidates (list): List of candidate descriptions
        args (argument class): Runtime arguments
        tokenizer (transformers Tokenizer): Tokenizer for the embedding_model
        embedding_model (transformer Model): Transformer model for embedding candidate descriptions

    Returns:
        feats (torch.tensor): Embeddings of the candidate descriptions
    """
    # Tokenize candidate descriptions
    feats = [tokenizer.encode_plus(val, add_special_tokens=True,max_length=args.max_candidate_len,
                                   padding='max_length', truncation='longest_first')
             for val in candidates]

    # Encode tokenized descriptions
    with torch.no_grad():
        feats = {key: torch.tensor([f[key] for f in feats]).to(embedding_model.device) for key in feats[0]}
        embedded_feats = embedding_model(**feats)  # [num_candidates, max_candidate_len, hidden_dim]

    # Reduce/pool descriptions embeddings if required
    if args.set_similarity:
        feats = embedded_feats.last_hidden_state.detach().cpu()  # [num_candidates, max_candidate_len, hidden_dim]
    elif args.candidate_pooling == 'cls':
        feats = embedded_feats.pooler_output.detach().cpu()  # [num_candidates, hidden_dim]
    elif args.candidate_pooling == "mean":
        feats = embedded_feats.last_hidden_state.detach().cpu()
        feats = feats.sum(1)
        feats = torch.nn.functional.layer_norm(feats, feats.size())
        feats = feats.detach().cpu()  # [num_candidates, hidden_dim]

    return feats


def get_slot_candidate_embeddings(ontology: dict, set_type: str, args, tokenizer, embedding_model, save_to_file=True):
    """
    Get embeddings for slots and candidates

    Args:
        ontology (dict): Dictionary of domain-slot pair descriptions and possible value sets
        set_type (str): Subset of the dataset being used (train/validation/test)
        args (argument class): Runtime arguments
        tokenizer (transformers Tokenizer): Tokenizer for the embedding_model
        embedding_model (transformer Model): Transormer model for embedding candidate descriptions
        save_to_file (bool): Indication of whether to save information to file

    Returns:
        slots (dict): domain-slot description embeddings, candidate embeddings and requestable flag for each domain-slot
    """
    # Set model to eval mode
    embedding_model.eval()

    slots = dict()
    for domain, subset in tqdm(ontology.items(), desc='Domains'):
        for slot, slot_info in tqdm(subset.items(), desc='Slots'):
            # Get description or use "domain-slot"
            if args.use_descriptions:
                desc = slot_info['description']
            else:
                desc = f"{domain}-{slot}"

            # Encode domain-slot pair description
            slot_emb = encode_candidates([desc], args, tokenizer, embedding_model)[0]

            # Obtain possible value set and discard requestable value
            values = deepcopy(slot_info['possible_values'])
            is_requestable = False
            if '?' in values:
                is_requestable = True
                values.remove('?')

            # Encode value candidates
            if values:
                feats = encode_candidates(values, args, tokenizer, embedding_model)
            else:
                feats = None

            # Store domain-slot description embeddings, candidate embeddings and requestabke flag for each domain-slot
            slots[f"{domain}-{slot}"] = (slot_emb, feats, is_requestable)

    # Dump tensors and ontology for use in training and evaluation
    if save_to_file:
        writer = os.path.join(args.output_dir, 'database', '%s.db' % set_type)
        torch.save(slots, writer)

        writer = open(os.path.join(args.output_dir, 'database', '%s.json' % set_type), 'w')
        json.dump(ontology, writer, indent=2)
        writer.close()
    
    return slots
