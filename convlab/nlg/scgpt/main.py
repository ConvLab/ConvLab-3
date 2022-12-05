import sys
sys.path.append('../../..')

import argparse
import json
from tqdm import tqdm
import time
import torch
from functools import reduce
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from transformers import get_linear_schedule_with_warmup

from convlab.util.unified_datasets_util import load_dataset, load_nlg_data, load_ontology
from convlab.nlg.scgpt.util import act2str
from convlab.nlg.scgpt.model import SCGPTDataset
from evaluate import GentScorer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from util import build_mask
from scgpt_special_tokens import *

code_test = False

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--train_ratio", default=1.0, type=float)
parser.add_argument("--accumulation_step", default=4, type=int)
parser.add_argument("--epoch_num", default=20, type=int)
parser.add_argument("--val_step", default=100, type=int)
parser.add_argument('--do_train', action="store_true", help="Whether to run training.")
parser.add_argument('--dataset', default="multiwoz21", type=str, help="The name of the dataset to be used.")
parser.add_argument('--model_path', default="", type=str, help="The path of model for testing.")
parser.add_argument('--base_model_name_path', default="gpt2", type=str, help="The path of base model.")
parser.add_argument('--scgpt_model_ckpt_path', default=None, type=str, help="The path of model for testing.")
parser.add_argument('--save_path', default="saved_models", type=str, help="Model save path.")
parser.add_argument('--exp_name', default="default_name", type=str, help="Current experiment name.")
parser.add_argument("--max_seq_len", default=128, type=int)
parser.add_argument("--save_epoch_interval", default=1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

# TensorBoard
tb_dir = 'runs/' + FLAGS.exp_name
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)
tb_writer = SummaryWriter(tb_dir, flush_secs=5)

## load model
if FLAGS.scgpt_model_ckpt_path is None:
    tokenizer = GPT2Tokenizer.from_pretrained(FLAGS.base_model_name_path)
    model = GPT2LMHeadModel.from_pretrained(FLAGS.base_model_name_path).to(local_rank)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(FLAGS.base_model_name_path)
    model = GPT2LMHeadModel(config=GPT2Config.from_pretrained(FLAGS.base_model_name_path)).to(local_rank)
    model.load_state_dict(torch.load(FLAGS.scgpt_model_ckpt_path))
    print('model load from ' + FLAGS.scgpt_model_ckpt_path)

nll_loss = nn.NLLLoss(reduce=False).to(local_rank)
ce_loss = nn.CrossEntropyLoss(reduce=False).to(local_rank)
def cal_loss(input, target, seq_lens, seq_lens_input):
    """Only calculate loss on responses, not on dialog act"""
    global nll_loss
    """Input: [batch, length, vocab]; target: [batch, length]; seq_lens: [batch]"""
    log_probs = F.log_softmax(input, dim=-1).transpose(1, 2)
    loss = nll_loss(log_probs, target)
    # loss = ce_loss(input, target)
    mask = build_mask(torch.max(seq_lens).item()-1, seq_lens-1).to(local_rank)
    input_mask = build_mask(torch.max(seq_lens).item()-1, seq_lens_input-1).to(local_rank)
    output_mask = torch.logical_xor(mask, input_mask)
    pad_mask = torch.logical_not(mask)
    masked_loss = loss * output_mask
    # masked_loss = loss * (output_mask + pad_mask)
    mean_loss = torch.sum(masked_loss) / torch.sum(output_mask + pad_mask)
    return mean_loss


def pad_collate(ori_batch):
    """
    Returns:
    batch: batch * max_len
    seq_lens: the length of len(da)+1+len(response)
    seq_lens_input: the length of len(da)
    """
    START_OF_PRED_ID = tokenizer._convert_token_to_id_with_added_voc('&')
    batch = [item[0] + [START_OF_PRED_ID] + item[1] + [tokenizer.eos_token_id] for item in ori_batch]
    output_lens = [len(item[1])+1 for item in ori_batch]
    batch = [item[-FLAGS.max_seq_len:] for item in batch]
    max_len = max([len(item) for item in batch])
    # print('max_len', max_len)
    seq_lens = [len(item) for item in batch]
    seq_lens_input = []
    for idx in range(len(batch)):
        curr_ipt_len = seq_lens[idx] - output_lens[idx]
        if curr_ipt_len < 0:
            curr_ipt_len = 0
        seq_lens_input.append(curr_ipt_len)
    batch = [item + [0]*(max_len-len(item)) for item in batch]
    return torch.LongTensor(batch), torch.LongTensor(seq_lens), torch.LongTensor(seq_lens_input)

## Training Hyper-params
def train(model, nlg_data, global_step=0):
    train_dataset = SCGPTDataset(filter_empty_nlg_data(nlg_data['train']), tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=2, sampler=train_sampler, collate_fn=pad_collate)

    val_dataset = SCGPTDataset(filter_empty_nlg_data(nlg_data['validation']), tokenizer)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, num_workers=2, sampler=val_sampler, collate_fn=pad_collate)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr)
    t_total = len(train_dataloader) * FLAGS.epoch_num // FLAGS.accumulation_step
    warm_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps,
                                                num_training_steps=t_total)
    model.train()
    for epoch in range(FLAGS.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)
        for batch_id, (inputs, seq_lens, seq_lens_input) in enumerate(tqdm(train_dataloader, desc=f'EPOCH:[{epoch+1}/{FLAGS.epoch_num}]')):
            if (batch_id+1) % FLAGS.accumulation_step == 0:
                global_step += 1
            inputs = inputs.to(local_rank)
            seq_lens = seq_lens.to(local_rank)
            seq_lens_input = seq_lens_input.to(local_rank)
            outputs = model(inputs, attention_mask=(inputs!=0).float())
            preds = outputs[0]
            loss = cal_loss(preds[:, :-1, :], inputs[:, 1:], seq_lens, seq_lens_input)
            loss /= FLAGS.accumulation_step
            loss /= dist.get_world_size() 
            loss.backward()
            # update params
            

            if (batch_id+1) % FLAGS.accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                # tensorboard
                if dist.get_rank() == 0:
                    tb_writer.add_scalar(f'Train/loss', loss.item(), global_step)
                    tb_writer.add_scalar(f'Train/PPL', torch.exp(loss).item(), global_step)
                    tb_writer.add_scalar(f'Train/Learning Rate', scheduler.get_last_lr()[0], global_step)
                if global_step % FLAGS.val_step == 0:
                    model.eval()
                    val_loss = eval(model, val_dataloader)
                    ppl = np.exp(val_loss)
                    if dist.get_rank() == 0:
                        tb_writer.add_scalar(f'Val/Loss', val_loss, global_step)
                        tb_writer.add_scalar(f'Val/PPL', ppl, global_step)
                    model.train()
            
        # save the model when each epoch ends
        if dist.get_rank() == 0:
            if (epoch+1) % FLAGS.save_epoch_interval == 0:
                # vaidation
                model.eval()
                val_loss = eval(model, val_dataloader)
                ppl = np.exp(val_loss)
                tb_writer.add_scalar(f'Val/Loss', val_loss, global_step)
                tb_writer.add_scalar(f'Val/PPL', ppl, global_step)
                model.train()
                # save model
                save_dir = os.path.join(FLAGS.save_path, FLAGS.exp_name, f'epoch_{epoch}')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(save_dir, f'epoch_{epoch}_step{global_step}.pt'))
                tokenizer.save_pretrained(save_dir)
                torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
                torch.save(scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pt'))
                print(f'Save model checkpoint to [{save_dir}]')
    tb_writer.flush()


def eval(model, loader, use_tqdm=False):
    with torch.no_grad():
        loss_list = []
        iter = tqdm(loader, desc='Val') if use_tqdm else loader
        for inputs, seq_lens, seq_lens_input in iter:
            inputs = inputs.to(local_rank)
            seq_lens = seq_lens.to(local_rank)
            seq_lens_input = seq_lens_input.to(local_rank)
            outputs = model(inputs)
            preds = outputs[0]
            loss = cal_loss(preds[:, :-1, :], inputs[:, 1:], seq_lens, seq_lens_input)
            loss_list.append(loss.item())
        mean_loss = np.mean(loss_list)
    return mean_loss


def inference_batch(model, sents):
    """Inference model given a batch of sents."""
    with torch.no_grad():
        sents = [sent + ' &' for sent in sents]
        sent_ids = [tokenizer.encode(sent) for sent in sents]
        max_len = max([len(sent) for sent in sent_ids])
        # ma_len = min(max_len, FLAGS.max_seq_len)
        sent_ids = [[0]*(max_len-len(sent)) + sent for sent in sent_ids]
        inputs = torch.LongTensor(sent_ids).to(local_rank)
        model_to_run = model.module if type(model) is DDP else model
        outputs = model_to_run.generate(inputs, attention_mask=(inputs != 0).float(), max_length=FLAGS.max_seq_len, eos_token_id=tokenizer.eos_token_id)  # greedy
        # outputs = model_to_run.generate(inputs, num_beams=4, max_length=513, eos_token_id=gpt2_tokenizer.eos_token_id,
        #                                 pad_token_id=gpt2_tokenizer.pad_token_id)  # beam search
        # output_strs = [tokenizer.decode(item) for item in outputs]
        outputs = outputs[:, len(inputs[0]):]
        output_strs = tokenizer.batch_decode(outputs)
        return output_strs


def inference_sent(model, sent):
    """Inference model given one single sentence."""
    return inference_batch(model, [sent])[0]


def inference_sents(model, sents):
    """Get the outputs of multiple sentences."""
    outputs = []
    for sent in tqdm(sents, desc='Inference Sentences'):
        output = inference_sent(model, sent)
        outputs.append(output)
    return outputs


def inference_sents_by_batch(model, sents):
    """Get the outputs of multiple sentences."""
    start_idx = 0
    ret = []
    start = time.time()
    while start_idx < len(sents):
        end_idx = start_idx + FLAGS.batch_size
        curr_sents = sents[start_idx:end_idx]
        outputs = inference_batch(model, curr_sents)
        ret += outputs
        start_idx += FLAGS.batch_size
        time_remain = (time.time()-start) / start_idx * (len(sents) - start_idx)
        print('{}/{}, time remaining: {:.2f}'.format(start_idx, len(sents), time_remain))
    return ret


def test(model, nlg_data, ontology, model_path):
    """将sheel中的GPU个数设为1运行"""
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f'model loaded from [{model_path}]')
    # Load test nlg data
    test_data = filter_empty_nlg_data(nlg_data['test'])
    dialog_acts = [act2str(item['dialogue_acts']).strip() for item in test_data]
    golden_responses = [item['utterance'].strip() for item in test_data]
    # dialog_acts = dialog_acts[:10]
    # golden_responses = golden_responses[:10]
    outputs = inference_sents_by_batch(model, dialog_acts)
    def get_real_output(ipt):
        if tokenizer.eos_token in ipt:
            ipt = ipt[:ipt.index(tokenizer.eos_token)].strip()
        return ipt
    outputs = [get_real_output(item) for item in outputs]
    if not os.path.exists('./test_outputs'):
        os.makedirs('./test_outputs', exist_ok=True)
    output_file = f'./test_outputs/{FLAGS.exp_name}.json'
    if dist.get_rank() == 0:
        with open(output_file, 'w+') as f:
            result = []
            for i in range(len(dialog_acts)):
                result.append({
                    'dialogue_acts': test_data[i]['dialogue_acts'],
                    'utterance': test_data[i]['utterance'],
                    'predictions': {
                        'utterance': outputs[i]
                    }
                })
            json.dump(result, f, indent=2, ensure_ascii=False)
    evaluator = GentScorer()
    parallel_corpus = []
    # BLEU
    for i in range(len(dialog_acts)):
        parallel_corpus.append([[golden_responses[i]], [outputs[i]]])
    BLEU_Score = evaluator.scoreSBLEU(parallel_corpus)
    # ERR
    ## all values in ontology
    val2ds_dict = {}
    for domain_name in ontology['domains']:
        domain = ontology['domains'][domain_name]
        for slot_name in domain['slots']:
            slot = domain['slots'][slot_name]
            if 'possible_values' not in slot:
                continue
            possible_vals = slot['possible_values']
            if len(possible_vals) > 0:
                for val in possible_vals:
                    val2ds_dict[val] = f'{domain_name}-{slot_name}'
    ## missing values
    score_list = []
    for item in test_data:
        da = item['dialogue_acts']
        utterance = item['utterance']
        missing_count = 0
        redundant_count = 0
        all_count = 0
        all_values = set()
        for key in da:
            slot_value = da[key]
            for triple in slot_value:
                if 'value' in triple:
                    value = triple['value']
                    all_values.add(value)
                    if value.strip().lower() not in utterance.lower():
                        missing_count += 1
                    all_count += 1
        if all_count == 0:
            continue
        ## redundant values
        for val in val2ds_dict:
            if f' {val.strip().lower()} ' in f' {utterance.strip().lower()} ' and val.strip().lower() not in all_values:
                redundant_count += 1
        item_score = float(redundant_count + redundant_count) / all_count
        score_list.append(item_score)
    ERR_Score = np.mean(score_list)
    print(f'BLEU: {BLEU_Score}\nERR_Score: {ERR_Score}')
    # with open(output_file, 'a') as f:
    #     f.write(f'BLEU: {BLEU_Score}\nERR_Score: {ERR_Score}')
    #     f.close()

def filter_empty_nlg_data(data):
    ret = []
    empty_number = 0
    for item in data:
        acts = item['dialogue_acts']
        acts_size = len(acts['binary']) + len(acts['categorical']) + len(acts['non-categorical'])
        if acts_size == 0:
            empty_number += 1
            continue
        else:
            ret.append(item)
    print('empty count: ', empty_number)
    return ret


if __name__ == '__main__':
    if '_' in FLAGS.dataset:
        spans = FLAGS.dataset.split('_')
        data_list = spans
        datasets = [load_dataset(item) for item in data_list] 
        nlg_datas = [load_nlg_data(item) for item in datasets]
        ret = {}
        def aggregrate(nlg_datas, split):
            ret = []
            for item in nlg_datas:
                ret += item[split]
            return ret
        ret['train'] = aggregrate(nlg_datas, 'train')
        ret['validation'] = aggregrate(nlg_datas, 'validation')
        ret['test'] = aggregrate(nlg_datas, 'test')
        if FLAGS.do_train:
            train(model, ret)
        else:
            print('not supported')
    else:
        dataset = load_dataset(FLAGS.dataset, dial_ids_order=0, split2ratio={'train': FLAGS.train_ratio})
        ontology = load_ontology(FLAGS.dataset)
        nlg_data = load_nlg_data(dataset)
        if FLAGS.do_train:
            train(model, nlg_data)
        else:
            test(model, nlg_data, ontology, FLAGS.model_path)
