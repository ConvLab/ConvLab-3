# -*- coding: utf-8 -*-
# Copyright 2020 DSML Group, Heinrich Heine University, Düsseldorf
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

import torch
import numpy as np


# Slot mapping table for description extractions
# SLOT_NAME_MAPPINGS = {
#     'arrive at': 'arriveAt',
#     'arrive by': 'arriveBy',
#     'leave at': 'leaveAt',
#     'leave by': 'leaveBy',
#     'arriveby': 'arriveBy',
#     'arriveat': 'arriveAt',
#     'leaveat': 'leaveAt',
#     'leaveby': 'leaveBy',
#     'price range': 'pricerange'
# }

# Set up global data directory
def set_datadir(dir):
    global DATA_DIR
    DATA_DIR = dir


# Set seeds
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# Get embeddings for slots and candidates
def get_slot_candidate_embeddings(set_type, args, tokenizer, embedding_model, save_to_file=True):
    # Get set alots and candidates
    reader = open(os.path.join(DATA_DIR, 'ontology_%s.json' % set_type), 'r')
    ontology = json.load(reader)
    reader.close()

    reader = open(os.path.join(DATA_DIR, 'slot_descriptions.json'), 'r')
    slot_descriptions = json.load(reader)
    reader.close()

    embedding_model.eval()

    slots = dict()
    for slot in ontology:
        if args.use_descriptions:
            # d, s = slot.split('-', 1)
            # s = SLOT_NAME_MAPPINGS[s] if s in SLOT_NAME_MAPPINGS else s
            # s = d + '-' + s
            # if slot in slot_descriptions:
            desc = slot_descriptions[slot]
            # elif slot.lower() in slot_descriptions:
            #     desc = slot_descriptions[s.lower()]
            # else:
            #     desc = slot.replace('-', ' ')
        else:
            desc = slot

        # Tokenize slot and get embeddings
        feats = tokenizer.encode_plus(desc, add_special_tokens = True,
                                            max_length = args.max_slot_len, padding='max_length',
                                            truncation = 'longest_first')

        with torch.no_grad():
            input_ids = torch.tensor([feats['input_ids']]).to(embedding_model.device) # [1, max_slot_len]
            if 'token_type_ids' in feats:
                token_type_ids = torch.tensor([feats['token_type_ids']]).to(embedding_model.device) # [1, max_slot_len]
                if 'attention_mask' in feats:
                    attention_mask = torch.tensor([feats['attention_mask']]).to(embedding_model.device) # [1, max_slot_len]
                    feats, pooled_feats = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
                    attention_mask = attention_mask.unsqueeze(-1).repeat((1, 1, feats.size(-1)))
                    feats = feats * attention_mask # [1, max_slot_len, hidden_dim]
                else:
                    feats, pooled_feats = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids)
            else:
                if 'attention_mask' in feats:
                    attention_mask = torch.tensor([feats['attention_mask']]).to(embedding_model.device)
                    feats, pooled_feats = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                    attention_mask = attention_mask.unsqueeze(-1).repeat((1, 1, feats.size(-1)))
                    feats = feats * attention_mask # [1, max_slot_len, hidden_dim]
                else:
                    feats, pooled_feats = embedding_model(input_ids=input_ids) # [1, max_slot_len, hidden_dim]
        
        if args.set_similarity:
            slot_emb = feats[0, :, :].detach().cpu() # [seq_len, hidden_dim]
        else:
            if args.candidate_pooling == 'cls' and pooled_feats is not None:
                slot_emb = pooled_feats[0, :].detach().cpu() # [hidden_dim]
            elif args.candidate_pooling == 'mean':
                feats = feats.sum(1)
                feats = torch.nn.functional.layer_norm(feats, feats.size())
                slot_emb = feats[0, :].detach().cpu() # [hidden_dim]

        # Tokenize value candidates and get embeddings
        values = ontology[slot]
        is_requestable = False
        if 'request' in values:
            is_requestable = True
            values.remove('request')
        if values:
            feats = [tokenizer.encode_plus(val, add_special_tokens = True,
                                                max_length = args.max_candidate_len, padding='max_length',
                                                truncation = 'longest_first')
                    for val in values]
            with torch.no_grad():
                input_ids = torch.tensor([f['input_ids'] for f in feats]).to(embedding_model.device) # [num_candidates, max_candidate_len]
                if 'token_type_ids' in feats[0]:
                    token_type_ids = torch.tensor([f['token_type_ids'] for f in feats]).to(embedding_model.device) # [num_candidates, max_candidate_len]
                    if 'attention_mask' in feats[0]:
                        attention_mask = torch.tensor([f['attention_mask'] for f in feats]).to(embedding_model.device) # [num_candidates, max_candidate_len]
                        feats, pooled_feats = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
                        attention_mask = attention_mask.unsqueeze(-1).repeat((1, 1, feats.size(-1)))
                        feats = feats * attention_mask # [num_candidates, max_candidate_len, hidden_dim]
                    else:
                        feats, pooled_feats = embedding_model(input_ids=input_ids, token_type_ids=token_type_ids) # [num_candidates, max_candidate_len, hidden_dim]
                else:
                    if 'attention_mask' in feats[0]:
                        attention_mask = torch.tensor([f['attention_mask'] for f in feats]).to(embedding_model.device)
                        feats, pooled_feats = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                        attention_mask = attention_mask.unsqueeze(-1).repeat((1, 1, feats.size(-1)))
                        feats = feats * attention_mask # [num_candidates, max_candidate_len, hidden_dim]
                    else:
                        feats, pooled_feats = embedding_model(input_ids=input_ids) # [num_candidates, max_candidate_len, hidden_dim]
            
            if args.set_similarity:
                feats = feats.detach().cpu() # [num_candidates, max_candidate_len, hidden_dim]
            else:
                if args.candidate_pooling == 'cls' and pooled_feats is not None:
                    feats = pooled_feats.detach().cpu()
                elif args.candidate_pooling == "mean":
                    feats = feats.sum(1)
                    feats = torch.nn.functional.layer_norm(feats, feats.size())
                    feats = feats.detach().cpu()
        else:
            feats = None
        slots[slot] = (slot_emb, feats, is_requestable)

    # Dump tensors for use in training
    if save_to_file:
        writer = os.path.join(args.output_dir, 'database', '%s.db' % set_type)
        torch.save(slots, writer)
    
    return slots
