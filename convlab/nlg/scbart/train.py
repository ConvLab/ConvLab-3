import random
import sys
import argparse
import logging
import json
import os
import glob
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# from evaluate import evaluator
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (BartTokenizer, BartForConditionalGeneration)

MODEL_CLASSES = {
    'bart': (BartTokenizer, BartForConditionalGeneration)
}

EMAP = {
        -1: 'conclude',
        0: 'neutral', 
        1: 'fearful', 
        2: 'dissatisfied', 
        3: 'apologetic', 
        4: 'abusive', 
        5: 'excited', 
        6: 'satisfied'
    }

sys_emo_dict = {
    0: 'neutral',
    1: 'compassionate',
    2: 'apologetic',
    3: 'enthusiastic',
    4: 'appreciative'
}

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, required=True, help="Enter an identifier for this experiment")
parser.add_argument('--task', type=str, default='nlg', help="Select task to train the plm on. (nlg, erc)")
parser.add_argument('--data_dir', type=str, default='.', help="Path to data")
parser.add_argument('--model_type', type=str, default='bart', help="Select plm. (bart, etc., to be implemented)")
parser.add_argument('--model_checkpoint', type=str, default="facebook/bart-base", help="Select plm. (bart, etc., to be implemented)")

parser.add_argument('--seed', type=int, default=42, metavar='S', help="Random seed (default: 42)")
parser.add_argument('--max_input_len', type=int, default=128, help="Max input sequence length")
parser.add_argument('--max_output_len', type=int, default=128, help="Max output sequence length")
parser.add_argument('--train_batch_size', type=int, default=8, help="Batch size for training")
parser.add_argument('--eval_batch_size', type=int, default=1, help="Batch size for evaluation")
parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")

parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for Adam optimiser")
parser.add_argument('--temperature', type=float, default=1.0, help="Learning rate for Adam optimiser")

parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimiser.")
parser.add_argument('--warmup_proportion', type=float, default=0.0, help="Proportion of warm-up steps in the schedular")

# formatting input. json action as basic. Experiment with json+emotion first
parser.add_argument("--semantic_action", action='store_true', help="Whether to use formatted semantic actions")
parser.add_argument("--emotion_transition", action='store_true', help="Whether to include emotion transition")
parser.add_argument('--context_size', type=int, default=0, help="Context window size. Default: 0")

parser.add_argument("--pretrain", action='store_true', help="Pretrain NLG with actions first, followed by emotionally conditional generation")
parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
parser.add_argument('--finetune_from', type=str, default=None, help="Specify custom pretrained model. e.g. scbart")
parser.add_argument('--train_data', type=str, default='train', help="Specify custom training data. default: train")

# input type
parser.add_argument("--vanilla", action='store_true', help="vanilla sc-bart")
parser.add_argument("--emowoz2", action='store_true', help="sc-bart + system conduct (semantic)")
parser.add_argument("--emowoz2_prev_user_utt", action='store_true', help="sc-bart + system conduct (semantic) + prev user utterance")
# parser.add_argument("--context", action='store_true', help="sc-bart + previous user utterance")
# parser.add_argument("--context_emo", action='store_true', help="sc-bart + previous user utterance + previous user emotion")
# parser.add_argument("--context_emo_trans", action='store_true', help="sc-bart + previous user utterance + user emotion transition")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

seed_all(args.seed)
exp_dir = args.exp_id
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

tokenizer_class, model_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class EmoWOZNLG(Dataset):
    def __init__(self, split, tokenizer, args):
        df = pd.read_csv(f'{args.data_dir}/{args.task}-{split}.csv')
        with open(f'{args.data_dir}/emowoz_2.0.json', 'r') as f:
            sys_conduct = json.load(f)

        # for developing
        # df = df.head(100)

        did, tid, src, tgt, uid, con = [], [], [], [], [], []
        for _, row in df.iterrows():
            if split not in ['test']:
                if 'DMAGE' in row['dialogue_id']:
                    continue
                if isinstance(row['actions'], float):
                    continue
            hist_str = row['history']
            history = json.loads(hist_str)

            if split not in ['test']:
                sys_emotions = [sys_conduct[f"{row['dialogue_id']}-{row['turn_id']}"]]
            else:
                sys_emotions = [0,1,2,3,4]
            for sys_emo in sys_emotions:
                if args.vanilla:
                    # text = f'{row["actions"]}'
                    text = f"The realisation of dialogue actions {row['actions']} in natural language is "
                elif args.emowoz2:
                    # text = f'{row["actions"]} | {sys_emo}'
                    text = f"The realisation of dialogue actions {row['actions']} in natural language with {sys_emo_dict[sys_emo]} conduct is "
                elif args.emowoz2_prev_user_utt:
                    text = f"Given the user request '{history[-1]}', the realisation of dialogue actions {row['actions']} in natural language with {sys_emo_dict[sys_emo]} conduct is "
                else:
                    print('error: no source format is specified')
                    exit()
                did.append(row['dialogue_id'])
                tid.append(row['turn_id'])
                src.append(text)
                tgt.append(row['target'])
                con.append(sys_emo)
                uid.append(f"{row['dialogue_id']}-{row['turn_id']}-{str(sys_emo)}")

        if split == 'train':
            training_samples = list(zip(did, tid, src, tgt, uid, con))
            random.shuffle(training_samples)
            did, tid, src, tgt, uid, con = zip(*training_samples)

        self.dial_id = did
        self.turn_id = tid
        self.source = src
        self.target = tgt
        self.unique_id = uid
        self.sample_conduct = con

        self.tokenizer = tokenizer
        self.max_input_len = args.max_input_len

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        source = str(self.source[item])
        target = str(self.target[item])
        conduct = self.sample_conduct[item]
        unique_id = str(self.unique_id[item])

        model_input = self.tokenizer.encode_plus(
            source, 
            add_special_tokens=True,
            max_length=self.max_input_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = tokenizer.encode(
                target, 
                add_special_tokens=True,
                max_length=self.max_input_len,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True)

        dial_id = str(self.dial_id[item])
        turn_id = str(self.turn_id[item])
        
        return {
            'source_ids': model_input["input_ids"].flatten(),
            'source': source,
            'source_mask': model_input["attention_mask"].flatten(),
            'target_ids': labels.flatten(),
            'target': target,
            # 'target_mask': labels["attention_mask"],
            'dialogue_id': dial_id,
            'turn_id': turn_id,
            'unique_id': unique_id,
            'conduct': conduct
        }

train_dataset = EmoWOZNLG(f'{args.train_data}', tokenizer, args)
print(len(train_dataset))
dev_dataset = EmoWOZNLG('dev', tokenizer, args)
print(len(dev_dataset))
test_dataset = EmoWOZNLG('test', tokenizer, args)
print(len(test_dataset))

# exit()
model = model_class.from_pretrained(args.model_checkpoint)

if args.finetune_from:
    print(f'loading checkpoint from {args.finetune_from}')
    model.load_state_dict(torch.load(args.finetune_from)['state_dict'])

skip_train = False
if os.path.exists(f'./{args.exp_id}/ckpt-0.pt'):
    print(f'loading checkpoint from ./{args.exp_id}/ckpt-best.pt. Skip training')
    model.load_state_dict(torch.load(f'./{args.exp_id}/ckpt-best.pt')['state_dict'])
    skip_train = True

model.to(args.device)

def compute_metrics(predictions, labels, tokenizer, metric):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["mean_gen_len"] = np.mean(prediction_lens)
    # result["mean_ref_len"] = np.mean()

    return result

def evaluate_model(args, eval_dataset, model, tokenizer):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    all_predictions, all_groundtruth = [], []
    all_generations, all_prompts, all_targets = [], [], []
    unique_ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch['source_ids'].to(args.device)
        labels = batch['target_ids'].to(args.device)
        all_prompts += batch['source']
        all_targets += batch['target']
        unique_ids += batch['unique_id']

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            eval_loss += loss.mean().item()
        eval_steps += 1

        output_ids = model.generate(inputs, num_beams=2, min_length=0, max_length=args.max_output_len, do_sample=True, temperature=args.temperature).cpu()
        nlg_outputs = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)
        all_generations += nlg_outputs
        all_predictions += output_ids
        all_groundtruth += labels.cpu()
    
    # bleu_evaluator = evaluator.load("bleu")
    # results = compute_metrics(all_predictions, all_groundtruth, tokenizer, bleu_evaluator)

    eval_loss /= eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    # print(perplexity)

    # results['perplexity'] = perplexity

    return perplexity.item(), {'unique_id': unique_ids, 'source': all_prompts, 'target': all_targets, 'generation': all_generations}

def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    total_steps = len(train_dataloader) * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps*args.warmup_proportion,
        num_training_steps=total_steps)

    model.resize_token_embeddings(len(tokenizer))

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    best_ppl = 10000000
    for e in train_iterator:
        for batch in tqdm(train_dataloader, desc="Training"):
            inputs = batch['source_ids'].to(args.device)
            masks = batch['source_mask'].to(args.device)
            labels = batch['target_ids'].to(args.device)

            model.train()
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # print(loss.item())

        state = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, f'{args.exp_id}/ckpt-{e}.pt')

        dev_result, dev_output = evaluate_model(args, dev_dataset, model, tokenizer)
        print(dev_result)
        pd.DataFrame(dev_output).to_csv(f'{args.exp_id}/dev-{e}.csv', index=False)

        if dev_result < best_ppl:
            print(f"Saving best model at Epoch {e}")
            # torch.save(state, f'{args.exp_id}/ckpt-best.pt')
            # best_ppl = dev_result
            model.save_pretrained(f'{args.exp_id}/ckpt-best.pt')
        else:
            if e > 0:
                print(f"Early stopping at Epoch {e}")
                break

if not skip_train:
    train(args, train_dataset, model)

test_result, test_output = evaluate_model(args, test_dataset, model, tokenizer)
print(test_result)
pd.DataFrame(test_output).to_csv(f'{args.exp_id}/test-best-temperate{args.temperature}.csv', index=False)