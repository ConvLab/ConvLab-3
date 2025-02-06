# coding=utf-8
#
# Copyright 2021 Heinrich Heine University Duesseldorf
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

import random
import sys
import argparse
import logging
import json
import math
import os
import glob
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.metrics import f1_score
import torch.nn.functional as F

from  torch.nn.modules.loss import _Loss

DEBUG = False

class EmotionDistanceLoss(torch.nn.Module):
    def __init__(self, device, balance='equal'):
        super(EmotionDistanceLoss, self).__init__()
        aspect_weight = [1,1,1] # valence, elicitor, conduct
        if balance != 'equal':
            if balance == 'v':
                aspect_weight = [2.8, 0.1, 0.1]
            if balance == 'e':
                aspect_weight = [0.1, 2.8, 0.1]
            if balance == 'c':
                aspect_weight = [0.1, 0.1, 2.8]
            if balance == 'v-e':
                aspect_weight = [1.4, 1.4, 0.2]
            if balance == 'v-c':
                aspect_weight = [1.4, 0.2, 1.4]
            if balance == 'e-c':
                aspect_weight = [0.2, 1.4, 1.4]
            if balance == 'v-e-c':
                aspect_weight = [1.5, 1, 0.5]
            if balance == 'v-c-e':
                aspect_weight = [1.5, 0.5, 1]
            if balance == 'e-v-c':
                aspect_weight = [1, 1.5, 0.5]
            if balance == 'e-c-v':
                aspect_weight = [0.5, 1.5, 1]
            if balance == 'c-v-e':
                aspect_weight = [1, 0.5, 1.5]
            if balance == 'c-e-v':
                aspect_weight = [0.5, 1, 1.5]
        aspect_distance = [
            [[0,0,0],   [1,0.5,0],  [1,0.5,0],  [1,0.5,0],  [1,0.5,1],  [1,0.5,0],  [1,0.5,0]   ],
            [[1,0.5,0], [0,0,0],    [0,1,0],    [0,1,0],    [0,1,1],    [2,0,0],    [2,1,0]     ],
            [[1,0.5,0], [0,1,0],    [0,0,0],    [0,1,0],    [0,0,1],    [2,1,0],    [2,0,0]     ],
            [[1,0.5,0], [0,1,0],    [0,1,0],    [0,0,0],    [0,1,1],    [2,1,0],    [2,1,0]     ],
            [[1,0.5,1], [0,1,1],    [0,0,1],    [0,1,1],    [0,0,0],    [2,1,1],    [2,0,1]     ],
            [[1,0.5,0], [2,0,0],    [2,1,0],    [2,1,0],    [2,1,1],    [0,0,0],    [0,1,0]     ],
            [[1,0.5,0], [2,1,0],    [2,0,0],    [2,1,0],    [2,0,1],    [0,1,0],    [0,0,0]     ]
        ]   # distance = aspect_distance[label, prediction]
        distance = np.zeros([7, 7])
        for row in range(len(aspect_distance)):
            for col in range(len(aspect_distance[0])):
                distance[row][col] = np.dot(aspect_weight, aspect_distance[row][col])
        # normalise
        normalised_distance = np.zeros([7, 7])
        for row in range(len(distance)):
            normalised_distance[row][:] = np.log(distance[row]+1)
        self.norm_d = torch.tensor(normalised_distance, dtype=torch.float).to(device)
    
    def forward(self, outputs, labels):
        # outputs: logits
        output_prob = F.softmax(outputs, dim=1)
        complementary_prob = 1 - output_prob
        complementary_logprob = torch.log(complementary_prob)
        
        # labels: indices of class label
        onehots = torch.tensor(F.one_hot(labels, 7), dtype=torch.float)

        # weights from the distance matrix
        weights = torch.matmul(onehots, self.norm_d)
        
        # row-wise dot-product between weights and complementary logprob

        loss = - (torch.sum(weights*complementary_logprob, dim=-1) / torch.sum(weights)) / 6

        # loss in a batch
        mean_loss = torch.mean(loss)
        return mean_loss


class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        if (input < 0).any() or (input > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')
        input = input + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(input * torch.log(input), dim=1))
        return H

def cross_entropy(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))


logger = logging.getLogger(__name__)

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

CLS_NUM = 7
DS_DIM = 361

EMAP = {'neutral': 0, 'fearful': 1, 'dissatisfied': 2, 'apologetic': 3, 'abusive': 4, 'excited': 5, 'satisfied': 6}
val_map = {0:0, 1:1, 2:1, 3:1, 4:1, 5:2, 6:2}   # 0: neutral, 1: negative, 2: positive
eli_map = {0:-1, 1:0, 2:1, 3:2, 4:1, 5:0, 6:1}    # -1: don't care, 0: event/fact, 1: operator, 2: self
con_map = {0:0, 1:0, 2:0, 3:0, 4:1, 5:0, 6:0} # 0: polite, 1: impolite
int_map = {'inform': 0, 'request': 1, 'start': 2, 'end': 3, 'book': -1, 'nobook': -1, 'empty': 4, 'system': -1}

class EmotionClassifier(nn.Module):
    def __init__(self, args):
        super(EmotionClassifier, self).__init__()
        self.config = BertConfig.from_json_file(f'{args.pretrained_model_dir}/config.json')
        self.bert = BertModel.from_pretrained(f'{args.pretrained_model_dir}/pytorch_model.bin', config=self.config)   # load bert

        n_classes = 7 if args.label_type == 'emotion' else 3
        self.drop = nn.Dropout(p=args.dropout_rate)   # define dropout
        self.takes_ds = args.dialog_state

        if args.dialog_state:
            if args.use_context:
                ds_input_dim = 3 * DS_DIM
            else:
                ds_input_dim = DS_DIM
            ds_output_dim = 256
            self.ds_projection = nn.Linear(ds_input_dim, ds_output_dim)
            self.tanh = nn.Tanh()
            feature_dim = self.bert.config.hidden_size + ds_output_dim
        else:
            feature_dim = self.bert.config.hidden_size

        self.out = nn.Linear(feature_dim, n_classes)   # linear layer for emotion classification
        self.out_valence = nn.Linear(feature_dim, 3)   # linear layer for valence classification
        self.out_elicitor = nn.Linear(feature_dim, 3)   # linear layer for elicitor classification
        self.out_conduct = nn.Linear(feature_dim, 2)   # linear layer for conduct classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, ds=None):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        if self.takes_ds:
            cls_feature = self.drop(pooled_output)
            ds_feature = self.tanh(self.ds_projection(ds))
            output = torch.cat((cls_feature, ds_feature), dim=1)
        else:
            output = self.drop(pooled_output)

        valence = self.out_valence(output)
        elicitor = self.out_elicitor(output)
        conduct = self.out_conduct(output)

        emotion = self.out(output)

        return emotion, valence, elicitor, conduct

class EmoWOZ(Dataset):
    def __init__(self, texts, labels, valence, elicitor, conduct, utt_ids, tokenizer, max_len, dialog_state):
        self.texts = texts
        self.labels = labels
        self.valence = valence
        self.elicitor = elicitor
        self.conduct = conduct
        self.utt_ids = utt_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dialog_state = dialog_state

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        utt_id = str(self.utt_ids[item])
        label = self.labels[item]
        valence = self.valence[item]
        elicitor = self.elicitor[item]
        conduct = self.conduct[item]
        dialog_state = self.dialog_state[item]
        
        # encoding.keys(): 'input_ids', 'attention_mask'
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'utt_id': utt_id,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'valence': torch.tensor(valence, dtype=torch.long),
            'elicitor': torch.tensor(elicitor, dtype=torch.long),
            'conduct': torch.tensor(conduct, dtype=torch.long),
            'dialog_state': torch.tensor(dialog_state, dtype=torch.float)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EmoWOZ(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        valence=df.valence.to_numpy(),
        elicitor=df.elicitor.to_numpy(),
        conduct=df.conduct.to_numpy(),
        utt_ids=df.utt_id.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        dialog_state=df.dialog_state.to_numpy()
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

def create_class_weight(train_set, label_type, mu):
    label_count = {}

    for i in range(7 if label_type == 'emotion' else 3):
        label_count[i] = 0
    
    for d in train_set:
        for l in d['label'].tolist():
            label_count[l] += 1

    total = np.sum(list(label_count.values()))
    weights = []
    for k in label_count:
        score = math.log(mu*total/label_count[k])
        weights.append(score)
    return weights

def aspect_to_emotion(v, e, c):
    if c == 1:
        return 4
    
    if v == 0:
        return 0

    if v == 1:
        if e == 0:
            return 1
        if e == 1:
            return 2
        if e == 2:
            return 3
    
    if v == 2:
        if e == 0 or e == 2:
            return 5
        if e == 1:
            return 6
    

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, args):
    model = model.train()
    losses = []
    distance_loss = EmotionDistanceLoss(device, balance=args.dloss_balance)

    for _, d in tqdm(enumerate(data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        emo_label = d["label"].to(device)
        val_label = d["valence"].to(device)
        eli_label = d["elicitor"].to(device)
        con_label = d["conduct"].to(device)
        dialog_states = d["dialog_state"].to(device)

        emo_out, val_out, eli_out, con_out = model(input_ids=input_ids, attention_mask=attention_mask, ds=dialog_states)

        loss = 0
        loss_count = 0
        if args.emotion:
            loss += 0.4* loss_fn(emo_out, emo_label)
            if args.distance_loss:
                loss = 0.4 * distance_loss(emo_out, emo_label)
            loss_count += 0.4
        if args.valence:
            loss += 0.2*loss_fn(val_out, val_label)
            loss_count += 0.2
        if args.elicitor:
            loss += 0.2*loss_fn(eli_out, eli_label)
            loss_count += 0.2
        if args.conduct:
            loss += 0.2*loss_fn(con_out, con_label)
            loss_count += 0.2
        
        loss = loss / loss_count
        losses.append(loss.item())
        # print(f'\rBatch number {i+1} out of {batch_num}: current mean loss = {np.mean(losses)}', end='', flush=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses)

def inference(model, data_loader, device, args):
    model = model.eval()
    utterances, utterance_ids = [], []
    all_emo, all_val, all_eli, all_con, mapped_emo = [], [], [], [], []
    all_emo_label, all_val_label, all_eli_label, all_con_label = [], [], [], []

    with torch.no_grad():
        for _, d in tqdm(enumerate(data_loader)):
            utt = d['text']
            utt_id = d['utt_id']
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            dialog_states = d["dialog_state"].to(device)

            emo_out, val_out, eli_out, con_out = model(input_ids=input_ids, attention_mask=attention_mask, ds=dialog_states)

            _, emo_pred = torch.max(emo_out, dim=1)
            _, val_pred = torch.max(val_out, dim=1)
            _, eli_pred = torch.max(eli_out, dim=1)
            _, con_pred = torch.max(con_out, dim=1)

            all_emo_label += d["label"].cpu().tolist()
            all_val_label += d["valence"].cpu().tolist()
            all_eli_label += d["elicitor"].cpu().tolist()
            all_con_label += d["conduct"].cpu().tolist()

            emo_p = emo_pred.cpu().tolist()
            val_p = val_pred.cpu().tolist()
            eli_p = eli_pred.cpu().tolist()
            con_p = con_pred.cpu().tolist()

            all_emo += emo_p
            all_val += val_p
            all_eli += eli_p
            all_con += con_p

            mapped = []
            for v, e, c in zip(val_p, eli_p, con_p):
                mapped.append(aspect_to_emotion(v, e, c))
            mapped_emo += mapped

            utterances += utt
            utterance_ids += utt_id

    label_set = [0,1,2,3,4,5,6] if args.label_type == 'emotion' else [0,1,2]
    score_0 = f1_score(all_emo_label, all_emo, labels=label_set[1:], average='micro')
    score_1 = f1_score(all_emo_label, all_emo, labels=label_set[1:], average='macro')
    score_2 = f1_score(all_emo_label, all_emo, labels=label_set[1:], average='weighted')
    score_3 = f1_score(all_emo_label, all_emo, labels=label_set, average='micro')
    score_4 = f1_score(all_emo_label, all_emo, labels=label_set, average='macro')
    score_5 = f1_score(all_emo_label, all_emo, labels=label_set, average='weighted')

    f1_scores = [score_0, score_1, score_2, score_3, score_4, score_5]
    print(f1_scores)

    df = pd.DataFrame({
        'label': all_emo_label, 
        'prediction': all_emo,
        'mapped_pred': mapped_emo,
        'valence_pred': all_val,
        'valence_label': all_val_label,
        'elicitor_pred': all_eli,
        'elicitor_label': all_eli_label,
        'conduct_pred': all_con,
        'conduct_label': all_con_label,
        'utt_id': utterance_ids, 
        'text': utterances})
        
    return f1_scores, df

def train(train_set, dev_set, test_set, device, args):
    logger.info("Initialising BERT")
    model = EmotionClassifier(args).to(device)        

    dev_results = [{}, {}, {}, {}, {}, {}]
    test_results = [{}, {}, {}, {}, {}, {}]

    logger.debug(f'parameter size: {sum([param.nelement() for param in model.parameters()])}')

    # training configurations
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # todo: potential bug here, to be tested

    total_steps = len(train_set) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps*args.warmup_proportion,
        num_training_steps=total_steps
    )

    if args.use_weight:
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(create_class_weight(train_set, args.label_type, args.mu)), ignore_index=-1).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    logger.info("Start training")
    for epoch in range(args.epochs):
        logger.info(f"Training for epoch {epoch + 1} out of {args.epochs}")
        train_loss = train_epoch(model, train_set, loss_fn, optimizer, device, scheduler, args)
        logger.info(f"Training loss: {train_loss}")

        logger.info(f"Evaluating dev set for epoch {epoch + 1} out of {args.epochs}")
        dev_f1_scores, dev_df = inference(model, dev_set, device, args)
        logger.info(f"Validation F1 scores: {dev_f1_scores}")

        logger.info(f"Evaluating test set for epoch {epoch + 1} out of {args.epochs}")
        test_f1_scores, test_df = inference(model, test_set, device, args)
        logger.info(f"Test F1 scores: {test_f1_scores}")

        for i, (df1, tf1) in enumerate(zip(dev_f1_scores, test_f1_scores)):
            dev_results[i][epoch] = df1
            test_results[i][epoch] = tf1
        
        state_epoch = epoch+start_epoch
        state = {
            'epoch': state_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, f'{args.exp_id}/ckpt-{epoch}.pt')
        #torch.save(model.state_dict(), f'{args.exp_id}/ckpt-{epoch}.bin')

        dev_df.to_csv(f'{args.exp_id}/ckpt-{epoch}-dev.csv', index=False)
        test_df.to_csv(f'{args.exp_id}/ckpt-{epoch}-test.csv', index=False)

    logger.info("Training Report")
    metric_names = [
        'MicroF1 w/o Neutral', 
        'MacroF1 w/o Neutral', 
        'WeightedF1 w/o Neutral', 
        'MicroF1 w/o Neutral', 
        'MacroF1 w/o Neutral', 
        'WeightedF1 w/o Neutral']

    for i, (dr, tr) in enumerate(zip(dev_results, test_results)):
        best_ckpt, best_dev_f1 = max(dr.items(), key=lambda k: k[1])
        best_test_f1 = tr[best_ckpt]
        logger.info(f"Best {metric_names[i]} appears in epoch {best_ckpt} - dev: {best_dev_f1}, test: {best_test_f1}")


def load_data(args):
    logger.info("Creating data loaders")
    with open(f'{args.data_dir}/data-split.json') as json_file:
        id_dict = json.load(json_file)

    if args.select_data == 'emowoz':
        train_ids = id_dict['train']['multiwoz'] + id_dict['train']['dialmage']
        dev_ids = id_dict['dev']['multiwoz'] + id_dict['dev']['dialmage']
    else:
        train_ids = id_dict['train'][args.select_data]
        dev_ids = id_dict['dev'][args.select_data]

    test_ids = id_dict['test']['multiwoz'] + id_dict['test']['dialmage']
    with open(f'{args.data_dir}/emowoz-multiwoz.json') as json_file:
        multiwoz = json.load(json_file)
    with open(f'{args.data_dir}/emowoz-dialmage.json') as json_file:
        dialmage = json.load(json_file)
    emowoz_dialogues = {**multiwoz, **dialmage}
    selected_ids = train_ids + dev_ids + test_ids
    dialogues = {dial_id: emowoz_dialogues[dial_id] for dial_id in selected_ids}

    dialog_state = {}
    if args.dialog_state:
        ds_file = f'{args.data_dir}/dialog_state_setsumbt.pkl'
        _, _, ds, _ = pickle.load(open(ds_file, 'rb'))
        if args.use_context:
            for k in ds:
                curr_dial_states = []
                for i, s in enumerate(ds[k]):
                    curr_ds = s#[0]
                    if i%2 == 0:
                        if i == 0:
                            pad_ds_1 = np.zeros(curr_ds.shape)
                            pad_ds_2 = np.zeros(curr_ds.shape)
                        elif i == 2:
                            pad_ds_1 = ds[k][i-2]
                            pad_ds_2 = np.zeros(curr_ds.shape)
                        else:
                            pad_ds_1 = ds[k][i-2]#[0]
                            pad_ds_2 = ds[k][i-4]#[0]
                        ds_with_history = np.concatenate((curr_ds, pad_ds_2, pad_ds_1), axis=None)
                    else:
                        ds_with_history = curr_ds
                    curr_dial_states.append(ds_with_history)
                dialog_state[k] = curr_dial_states
        else:
            dialog_state = ds
    else:   # create dummy ds
        for k in dialogues.keys():
            curr_dial_states = []
            for _ in dialogues[k]['log']:
                if args.use_context:
                    curr_dial_states.append(np.zeros((DS_DIM*3,)))
                else:
                    curr_dial_states.append(np.zeros((DS_DIM,)))
            dialog_state[k] = curr_dial_states

    if args.use_context:
        for k in dialogues.keys():
            dial = [d['text'] for d in dialogues[k]['log']]
            for i in reversed(range(len(dial))):
                if i%2 == 0:
                    full_history = []
                    full_history[:] = dial[:i+1]
                    concat_str = ""
                    history = deepcopy(full_history)
                    for j in reversed(range(len(full_history))):  # reverse order to place the current turn closer to the [CLS]
                        if j%2 == 0:
                            concat_str += f"user: {history[j]} "   
                        else:
                            concat_str += f"system: {history[j]} "
                    dialogues[k]['log'][i]['text'] = concat_str

    train_utt, dev_utt, test_utt = [], [], []
    train_lab, dev_lab, test_lab = [], [], []
    train_utt_ids, dev_utt_ids, test_utt_ids = [], [], []
    train_ds, dev_ds, test_ds = [], [], []

    for k in train_ids:
        train_utt += [a['text'] for a in dialogues[k]['log'][0::2]]
        train_lab += [a['emotion'][3][args.label_type] for a in dialogues[k]['log'][0::2]]
        train_utt_ids += [k for _ in dialogues[k]['log'][0::2]]
        train_ds += dialog_state[k][0::2]
    for k in dev_ids:
        dev_utt += [a['text'] for a in dialogues[k]['log'][0::2]]
        dev_lab += [a['emotion'][3][args.label_type] for a in dialogues[k]['log'][0::2]]
        dev_utt_ids += [k for _ in dialogues[k]['log']][0::2]
        dev_ds += dialog_state[k][0::2]
    for k in test_ids:
        test_utt += [a['text'] for a in dialogues[k]['log'][0::2]]
        test_lab += [a['emotion'][3][args.label_type] for a in dialogues[k]['log'][0::2]]
        test_utt_ids += [k for _ in dialogues[k]['log'][0::2]]
        test_ds += dialog_state[k][0::2]

    train_valence = [val_map[i] for i in train_lab]
    train_elicitor = [eli_map[i] for i in train_lab]
    train_conduct = [con_map[i] for i in train_lab]

    dev_valence = [val_map[i] for i in dev_lab]
    dev_elicitor = [eli_map[i] for i in dev_lab]
    dev_conduct = [con_map[i] for i in dev_lab]

    aug_utt, aug_lab, aug_diff, aug_utt_ids, aug_val, aug_eli, aug_con, aug_ds = [], [], [], [], [], [], [], []
    if args.augment is not None:
        def aug_emotion(E, src):
            suffix = 'clean'
            if src != 'chitchat':
                suffix = 'to-inferred'

            aug_e = EMAP[E]
            aug_base_dir = f'{args.data_dir}'
            if aug_e == 0:
                augfile = ['neutral-goemo']
            if aug_e == 1:
                augfile = [f'fearful-{suffix}'] # ['fearful-goemo', 'fearful-ed']
            if aug_e == 2:
                augfile = ['dissatisfied-ed'] # ['dissatisfied-goemo', 'dissatisfied-ed']
            if aug_e == 3:
                augfile = [f'apologetic-{suffix}']
            if aug_e == 4:
                augfile = ['abusive-convabu']
            if aug_e == 5:
                augfile = [f'excited-{suffix}'] # ['excited-goemo', 'excited-ed']
            if aug_e == 6:
                augfile = ['satisfied-ed'] # ['satisfied-goemo', 'satisfied-ed']
            aug_txt = []
            for af in augfile:
                with open(f"{aug_base_dir}/{af}") as f:
                    aug_txt += f.read().splitlines()

            # pointers to all relevant emotions
            ctx_ptr = []
            for i in range(len(train_utt)):
                if train_lab[i] == aug_e:
                    ctx_ptr.append(i)
            temp_utt, temp_lab, temp_utt_ids, temp_val, temp_eli, temp_con, temp_ds = [], [], [], [], [], [], []
            for t in aug_txt:
                ptr = random.choice(ctx_ptr)
                if args.use_context:
                    ctx_utt = train_utt[ptr]
                    if aug_e != 4 and src != 'chitchat':
                        temp_utt.append(t)
                    else:
                        if ctx_utt.find('system: ') == -1:
                            temp_utt.append(f"user: {t}")
                        else:
                            temp_utt.append(f"user: {t} {ctx_utt[ctx_utt.find('system: '):]}")
                        
                else:
                    temp_utt.append(t)
                temp_lab.append(train_lab[ptr])
                temp_utt_ids.append(train_utt_ids[ptr])
                temp_val.append(train_valence[ptr])
                temp_eli.append(train_elicitor[ptr])
                temp_con.append(train_conduct[ptr])
                temp_ds.append(train_ds[ptr])
            return temp_utt, temp_lab, temp_utt_ids, temp_val, temp_eli, temp_con, temp_ds
        
        for augmented_emotion in ['abusive']:   # plug in abusive utterances only
            U, L, UI, V, E, C, D = aug_emotion(augmented_emotion, src=args.augment_src)
            aug_utt += U
            aug_lab += L
            aug_utt_ids += UI
            aug_val += V
            aug_eli += E
            aug_con += C
            aug_ds += D

            logger.info(f"Augment emotion {augmented_emotion} from chitchat dialogues. Number of utterances: {len(U)}")

    train_utt += aug_utt
    train_lab += aug_lab
    train_utt_ids += aug_utt_ids
    train_valence += aug_val
    train_elicitor += aug_eli
    train_conduct += aug_con
    train_ds += aug_ds

    temp_utt, temp_lab, temp_utt_ids, temp_val, temp_eli, temp_con, temp_ds = [], [], [], [], [], [], []
    if args.augment is not None:    # augment with task-oriented dialogues
        augment = pickle.load(open(f'{args.data_dir}/aug_with_ds_setsumbt.pkl', 'rb'))

        for k in augment:
            for t in augment[k]['emotion']:
                if args.use_context:
                    aug_str = augment[k]['emotion'][t]['ctxbert']
                    temp_utt.append(aug_str)
                else:
                    utt = augment[k]['emotion'][t]['ctxbert']
                    def find_nth(haystack, needle, n):
                        start = haystack.find(needle)
                        while start >= 0 and n > 1:
                            start = haystack.find(needle, start+len(needle))
                            n -= 1
                        return start
                    aug_str = utt[find_nth(utt, "user: ", 1)+6:find_nth(utt, ": ", 2)]
                    temp_utt.append(aug_str)
                temp_lab.append(augment[k]['emotion'][t]['label'])
                temp_utt_ids.append(k)
                temp_val.append(val_map[augment[k]['emotion'][t]['label']])
                temp_eli.append(eli_map[augment[k]['emotion'][t]['label']])
                temp_con.append(con_map[augment[k]['emotion'][t]['label']])
                if args.use_context:
                    curr_ds = augment[k]['dialog_state']['state_vector'][t][0]
                    if t >= 0 and t < 2:
                        pad_ds_1 = np.zeros(curr_ds.shape)
                        pad_ds_2 = np.zeros(curr_ds.shape)
                    elif t >=2 and t < 4:
                        pad_ds_1 = np.zeros(curr_ds.shape)
                        pad_ds_2 = augment[k]['dialog_state']['state_vector'][t-2][0]
                    else:
                        pad_ds_1 = augment[k]['dialog_state']['state_vector'][t-4][0]
                        pad_ds_2 = augment[k]['dialog_state']['state_vector'][t-2][0]
                    ds_with_history = np.concatenate((curr_ds, pad_ds_2, pad_ds_1), axis=None)
                    temp_ds.append(ds_with_history)
                else:
                    temp_ds.append(augment[k]['dialog_state']['state_vector'][t][0])

    train_utt += temp_utt
    train_lab += temp_lab
    train_utt_ids += temp_utt_ids
    train_valence += temp_val
    train_elicitor += temp_eli
    train_conduct += temp_con
    train_ds += temp_ds

    # shuffle training data
    train_data = list(zip(train_utt, train_lab, train_utt_ids, train_valence, train_elicitor, train_conduct, train_ds))
    
    random.shuffle(train_data)
    train_utt, train_lab, train_utt_ids, train_valence, train_elicitor, train_conduct, train_ds = zip(*train_data)

    df_train = pd.DataFrame({
        'text': train_utt, 
        'label': train_lab, 
        'utt_id': train_utt_ids, 
        'valence': train_valence,
        'elicitor': train_elicitor,
        'conduct': train_conduct, 
        'dialog_state': train_ds})

    df_dev = pd.DataFrame({
        'text': dev_utt, 
        'label': dev_lab, 
        'utt_id': dev_utt_ids,
        'valence': dev_valence,
        'elicitor': dev_elicitor,
        'conduct': dev_conduct, 
        'dialog_state': dev_ds})

    df_test = pd.DataFrame({
        'text': test_utt, 
        'label': test_lab, 
        'utt_id': test_utt_ids,
        'valence': [val_map[i] for i in test_lab],
        'elicitor': [eli_map[i] for i in test_lab],
        'conduct': [con_map[i] for i in test_lab], 
        'dialog_state': test_ds})

    if DEBUG:
        print(df_train.shape, df_dev.shape, df_test.shape)

    tokenizer = BertTokenizer(f'{args.pretrained_model_dir}/vocab.txt')
    
    if DEBUG:
        df_train = df_train.head(50)
        df_dev = df_dev.head(50)
        df_test = df_test.head(50)

    train_data_loader = create_data_loader(df_train, tokenizer, args.max_len, args.batch_size)
    dev_data_loader = create_data_loader(df_dev, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, args.max_len, args.batch_size)

    return train_data_loader, dev_data_loader, test_data_loader

def classify(device, args):
    _, _, classify_data_loader = load_data(args)
    logger.info("Finished loading test set") 
    
    model = EmotionClassifier(args)
    logger.info("Testing the selected model")
    model.load_state_dict(torch.load(f'{args.model_checkpoint}')['state_dict'])
    model = model.to(device)
    logger.info(f"Evaluating data with checkpoint {args.model_checkpoint}")
    infer_f1_scores, infer_df = inference(model, classify_data_loader, device, args)
    logger.info(f"Test F1 scores: {infer_f1_scores}")

    infer_id = 'test' 
    infer_df.to_csv(f'{args.exp_id}/{infer_id}-inference.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True, help="Enter an identifier for this experiment")
    parser.add_argument('--resume', type=str, help="Enter the ckeckpoint (model and optimizer state) to resume training")
    parser.add_argument('--select_data', type=str, default='emowoz', choices=['emowoz', 'multiwoz', 'dialmage'], help="Select data source for training (default: emowoz)")
    parser.add_argument('--label_type', type=str, default='emotion', choices=['emotion', 'sentiment'], help="Choose the type of label (default: emotion)")
    parser.add_argument('--data_dir', type=str, default='./data', help="Enter the path to the data directory")
    parser.add_argument('--pretrained_model_dir', type=str, default='./bert', help="Enter the path to the pretrained model directory")

    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help="Input batch size for training (default: 32)")
    parser.add_argument('--max_len', type=int, default=128, help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument('--epochs', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=42, metavar='S', help="Random seed (default: 42)")    
    parser.add_argument('--eval_criterion', type=int, default=1, choices=[0,1,2,3,4,5], 
        help="The criterion for the best model [0: micro f1 no neutral, 1: macro f1 no neutral, 2: weighted f1 no neutral, 3: micro f1 with neutral, 4: macro f1 with neutral, 5: weighted f1 with neutral] (default: 1)")

    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for Adam optimiser")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimiser.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for BERT representations.")
    parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument('--mu', type=float, default=1.0, help='Mu for weighted loss')

    parser.add_argument("--debug", action='store_true', help="Whether to run a small batch to debug")
    parser.add_argument("--do_train", action='store_true', help="Whether to run the training")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--use_context", action='store_true', help="Whether to use the entire dialogue history for classification")
    parser.add_argument("--do_lower_case", action='store_true', help="Whether to use an uncased model.")
    parser.add_argument("--keep_all_checkpoints", action='store_true', help="Whether to keep model checkpoints for all epochs during training (default: only keep the best model)")
    parser.add_argument("--use_weight", action='store_true', help="Whether to use weighted loss")

    parser.add_argument("--silent", action="store_const", dest="log_level", const=logging.NOTSET, default=logging.INFO, help="Whether to log messages")

    parser.add_argument("--emotion", action='store_true', help="Whether to train a emotion classification head")
    parser.add_argument("--valence", action='store_true', help="Whether to train a valence classification head")
    parser.add_argument("--elicitor", action='store_true', help="Whether to train a elicitor classification head")
    parser.add_argument("--conduct", action='store_true', help="Whether to train a conduct classification head")
    parser.add_argument("--dialog_state", action='store_true', help="Whether to use dialog state as auxiliary feature")
    parser.add_argument('--augment', nargs='+', help='emotions to augment')

    parser.add_argument("--do_classify", action='store_true', help="Whether to classify a set of utterances")
    parser.add_argument('--model_checkpoint', type=str, help="The model checkpoint to use for inference")

    parser.add_argument('--distance_loss', action='store_true', help="Use distance-based loss to penalise prediction of distant emotions")

    parser.add_argument('--batch_curriculum', type=str, default='none', choices=['none', 'easyfirst', 'hardfirst'], 
        help="Curriculum learning batch scheduler")
    parser.add_argument('--task_curriculum', type=str, default='none', choices=['none', 'valencefirst'], 
        help="Curriculum learning batch scheduler")

    parser.add_argument('--augment_src', type=str, default='chitchat', choices=['chitchat', 'to-inferred'], help="source of augmented data")
    parser.add_argument('--dloss_balance', type=str, default='equal', help="relative importance of three aspects in calculating the distance-based loss")

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.debug:
        DEBUG = True

    seed_all(args.seed)

    exp_dir = args.exp_id
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    logging.basicConfig(filename=f'{exp_dir}/log.txt', filemode='a', format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level=args.log_level)

    logger.info("Usage: {0}".format(" ".join([x for x in sys.argv])))

    if args.use_weight and args.mu <= 0:
        logger.info(f"Invalid value of mu ({args.mu}). Use default value 1.0")
        args.mu = 1.0

    logger.info("Setting:") 
    for k, v in sorted(vars(args).items()): 
        logger.info("{0}: {1}".format(k, v))

    train_dataloader, dev_dataloader, test_dataloader = load_data(args)
    logger.info("Data loading finished") 
    logger.info(f"Run training on {args.select_data}")
    
    if args.model_checkpoint is None:
        train(train_dataloader, dev_dataloader, test_dataloader, device, args)
        args.model_checkpoint = f'{args.exp_id}/best_model.pt'

    classify(device, args)

