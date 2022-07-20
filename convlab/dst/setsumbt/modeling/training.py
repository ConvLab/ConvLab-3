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
"""Training an evaluation utils"""

import random
import os
import logging
from copy import deepcopy

import torch
from torch.nn import DataParallel
from torch.distributions import Categorical
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
try:
    from apex import amp
except:
    print('Apex not used')

from convlab.dst.setsumbt.utils import clear_checkpoints
from convlab.dst.setsumbt.modeling import LinearTemperatureScheduler


# Load logger and tensorboard summary writer
def set_logger(logger_, tb_writer_):
    global logger, tb_writer
    logger = logger_
    tb_writer = tb_writer_


# Set seeds
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    logger.info('Seed set to %d.' % args.seed)


def set_ontology_embeddings(model, slots, load_slots=True):
    # Get slot and value embeddings
    values = {slot: slots[slot][1] for slot in slots}

    # Load model ontology
    if load_slots:
        slots = {slot: embs for slot, embs in slots.items()}
        model.add_slot_candidates(slots)
    for slot in model.informable_slot_ids:
        model.add_value_candidates(slot, values[slot], replace=True)


def train(args, model, device, train_dataloader, dev_dataloader, slots, slots_dev, embeddings=None, tokenizer=None):
    """Train model!"""

    # Calculate the total number of training steps to be performed
    if args.max_training_steps > 0:
        t_total = args.max_training_steps
        args.num_train_epochs = args.max_training_steps // (
            (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs

    if args.save_steps <= 0:
        args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps

    # Group weight decay and no decay parameters in the model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr":args.learning_rate
        },
    ]

    # Initialise the optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Initialise linear lr scheduler
    num_warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)

    # Initialise distillation temp scheduler
    if model.config.loss_function in ['distillation']:
        temp_scheduler = TemperatureScheduler(total_steps=t_total, base_temp=args.annealing_base_temp,
                                              cycle_len=args.annealing_cycle_len)
    else:
        temp_scheduler = None

    # Set up fp16 and multi gpu usage
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if args.n_gpu > 1:
        model = DataParallel(model)

    # Load optimizer checkpoint if available
    best_model = {'joint goal accuracy': 0.0,
                  'request f1 score': 0.0,
                  'active domain f1 score': 0.0,
                  'goodbye act f1 score': 0.0,
                  'train loss': np.inf}
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')):
        logger.info("Optimizer loaded from previous run.")
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))
        if temp_scheduler is not None:
            temp_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'temp_scheduler.pt')))
        if args.fp16 and os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')):
            logger.info("FP16 Apex Amp loaded from previous run.")
            amp.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'amp.pt')))

        # Evaluate initialised model
        if args.do_eval:
            # Set up model for evaluation
            model.eval()
            set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots_dev, load_slots=False)

            jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss, stats = train_eval(args, model, device, dev_dataloader)

            # Set model back to training mode
            model.train()
            model.zero_grad()
            set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots, load_slots=False)
        else:
            jg_acc, req_f1, dom_f1, bye_f1 = 0.0, 0.0, 0.0, 0.0

        best_model['joint goal accuracy'] = jg_acc
        best_model['request f1 score'] = req_f1
        best_model['active domain f1 score'] = dom_f1
        best_model['goodbye act f1 score'] = bye_f1

    # Log training set up
    logger.info("Device: %s, Number of GPUs: %s, FP16 training: %s" % (device, args.n_gpu, args.fp16))
    logger.info("***** Running training *****")
    logger.info("  Num Batches = %d" % len(train_dataloader))
    logger.info("  Num Epochs = %d" % args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d" % t_total)

    # Initialise training parameters
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d" % epochs_trained)
            logger.info("  Continuing training from global step %d" % global_step)
            logger.info("  Will skip the first %d steps in the first epoch" % steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    # Prepare model for training
    tr_loss, logging_loss = 0.0, 0.0
    model.train()
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    steps_since_last_update = 0
    # Perform training
    for e in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        # Iterate over all batches
        for step, batch in enumerate(epoch_iterator):
            # Skip batches already trained on
            if step < steps_trained_in_current_epoch:
                continue

            # Extract all label dictionaries from the batch
            input_dict = {}
            if 'goodbye_belief' in batch:
                input_dict['inform_labels'] = {slot: batch['belief-' + slot].to(device)
                                               for slot in model.informable_slot_ids if ('belief-' + slot) in batch}
                input_dict['request_labels'] = {slot: batch['request_belief-' + slot].to(device)
                                            for slot in model.requestable_slot_ids
                                            if ('request_belief-' + slot) in batch} if args.predict_actions else None
                input_dict['domain_labels'] = {domain: batch['domain_belief-' + domain].to(device)
                                            for domain in model.domain_ids
                                            if ('domain_belief-' + domain) in batch} if args.predict_actions else None
                input_dict['goodbye_labels'] = batch['goodbye_belief'].to(device) if args.predict_actions else None
            else:
                input_dict['inform_labels'] = {slot: batch['labels-' + slot].to(device)
                                               for slot in model.informable_slot_ids if ('labels-' + slot) in batch}
                input_dict['request_labels'] = {slot: batch['request-' + slot].to(device)
                                                for slot in model.requestable_slot_ids
                                                if ('request-' + slot) in batch} if args.predict_actions else None
                input_dict['domain_labels'] = {domain: batch['active-' + domain].to(device)
                                               for domain in model.domain_ids
                                               if ('active-' + domain) in batch} if args.predict_actions else None
                input_dict['goodbye_labels'] = batch['goodbye'].to(device) if args.predict_actions else None

            # Extract all model inputs from batch
            input_dict['input_ids'] = batch['input_ids'].to(device)
            input_dict['token_type_ids'] = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            input_dict['attention_mask'] = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

            # Set up temperature scaling for the model
            if temp_scheduler is not None:
                model.temp = temp_scheduler.temp()

            # Forward pass to obtain loss
            loss, _, _, _, _, _, stats = model(**input_dict)

            if args.n_gpu > 1:
                loss = loss.mean()

            # Update step
            if step % args.gradient_accumulation_steps == 0:
                loss = loss / args.gradient_accumulation_steps
                if temp_scheduler is not None:
                    tb_writer.add_scalar('Temp', temp_scheduler.temp(), global_step)
                tb_writer.add_scalar('Loss/train', loss, global_step)
                # Backpropogate accumulated loss
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        tb_writer.add_scalar('Scaled_Loss/train', scaled_loss, global_step)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Get learning rate
                lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('LearningRate', lr, global_step)

                if stats:
                    # print(stats.keys())
                    for slot, stats_slot in stats.items():
                        for key, item in stats_slot.items():
                            tb_writer.add_scalar(f'{key}_{slot}/Train', item, global_step)

                # Update model parameters
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if temp_scheduler is not None:
                    temp_scheduler.step()

                tr_loss += loss.float().item()
                epoch_iterator.set_postfix(loss=loss.float().item())
                loss = 0.0
                global_step += 1

            # Save model checkpoint
            if global_step % args.save_steps == 0:
                logging_loss = tr_loss - logging_loss

                # Evaluate model
                if args.do_eval:
                    # Set up model for evaluation
                    model.eval()
                    set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots_dev, load_slots=False)

                    jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss, stats = train_eval(args, model, device, dev_dataloader)
                    # Log model eval information
                    if req_f1 is not None:
                        logger.info('%i steps complete, Loss since last update = %f, Dev Joint goal acc = %f, Dev Slot acc = %f, Dev Request F1 Score = %f, Dev Domain F1 Score = %f, Dev Goodbye F1 Score = %f'
                                    % (global_step, logging_loss / args.save_steps, jg_acc, sl_acc, req_f1, dom_f1, bye_f1))
                        tb_writer.add_scalar('JointGoalAccuracy/Dev', jg_acc, global_step)
                        tb_writer.add_scalar('SlotAccuracy/Dev', sl_acc, global_step)
                        tb_writer.add_scalar('RequestF1Score/Dev', req_f1, global_step)
                        tb_writer.add_scalar('DomainF1Score/Dev', dom_f1, global_step)
                        tb_writer.add_scalar('GoodbyeF1Score/Dev', bye_f1, global_step)
                    else:
                        logger.info('%i steps complete, Loss since last update = %f, Dev Joint goal acc = %f, Dev Slot acc = %f'
                                    % (global_step, logging_loss / args.save_steps, jg_acc, sl_acc))
                        tb_writer.add_scalar('JointGoalAccuracy/Dev', jg_acc, global_step)
                        tb_writer.add_scalar('SlotAccuracy/Dev', sl_acc, global_step)
                    tb_writer.add_scalar('Loss/Dev', loss, global_step)
                    if stats:
                        for slot, stats_slot in stats.items():
                            for key, item in stats_slot.items():
                                tb_writer.add_scalar(f'{key}_{slot}/Dev', item, global_step)

                    # Set model back to training mode
                    model.train()
                    model.zero_grad()
                    set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots, load_slots=False)
                else:
                    jg_acc, req_f1 = 0.0, None
                    logger.info('%i steps complete, Loss since last update = %f' % (global_step, logging_loss / args.save_steps))

                logging_loss = tr_loss

                # Compute the score of the best model
                try:
                    best_score = (best_model['request f1 score'] * model.config.user_request_loss_weight) + \
                        (best_model['active domain f1 score'] * model.config.active_domain_loss_weight) + \
                        (best_model['goodbye act f1 score'] *
                         model.config.user_general_act_loss_weight)
                except AttributeError:
                    best_score = 0.0
                best_score += best_model['joint goal accuracy']

                # Compute the score of the current model
                try:
                    current_score = (req_f1 * model.config.user_request_loss_weight) + \
                        (dom_f1 * model.config.active_domain_loss_weight) + \
                        (bye_f1 * model.config.user_general_act_loss_weight) if req_f1 is not None else 0.0
                except AttributeError:
                    current_score = 0.0
                current_score += jg_acc

                # Decide whether to update the model
                if best_model['joint goal accuracy'] < jg_acc and jg_acc > 0.0:
                    update = True
                elif current_score > best_score and current_score > 0.0:
                    update = True
                elif best_model['train loss'] > (tr_loss / global_step) and best_model['joint goal accuracy'] == 0.0:
                    update = True
                else:
                    update = False

                if update:
                    steps_since_last_update = 0
                    logger.info('Model saved.')
                    best_model['joint goal accuracy'] = jg_acc
                    if req_f1:
                        best_model['request f1 score'] = req_f1
                        best_model['active domain f1 score'] = dom_f1
                        best_model['goodbye act f1 score'] = bye_f1
                    best_model['train loss'] = tr_loss / global_step

                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    if args.n_gpu > 1:
                        model.module.save_pretrained(output_dir)
                    else:
                        model.save_pretrained(output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    if temp_scheduler is not None:
                        torch.save(temp_scheduler.state_dict(), os.path.join(output_dir, 'temp_scheduler.pt'))
                    if args.fp16:
                        torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))

                    # Remove older training checkpoints
                    clear_checkpoints(args.output_dir, args.keep_models)
                else:
                    steps_since_last_update += 1
                    logger.info('Model not saved.')

            # Stop training after max training steps or if the model has not updated for too long
            if args.max_training_steps > 0 and global_step > args.max_training_steps:
                epoch_iterator.close()
                break
            if args.patience > 0 and steps_since_last_update >= args.patience:
                epoch_iterator.close()
                break

        logger.info('Epoch %i complete, average training loss = %f' % (e + 1, tr_loss / global_step))

        if args.max_training_steps > 0 and global_step > args.max_training_steps:
            train_iterator.close()
            break
        if args.patience > 0 and steps_since_last_update >= args.patience:
            train_iterator.close()
            logger.info('Model has not improved for at least %i steps. Training stopped!' % args.patience)
            break

    # Evaluate final model
    if args.do_eval:
        model.eval()
        set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots_dev, load_slots=False)

        jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss, stats = train_eval(args, model, device, dev_dataloader)
        if req_f1 is not None:
            logger.info('Training complete, Training Loss = %f, Dev Joint goal acc = %f, Dev Slot acc = %f, Dev Request F1 Score = %f, Dev Domain F1 Score = %f, Dev Goodbye F1 Score = %f'
                        % (tr_loss / global_step, jg_acc, sl_acc, req_f1, dom_f1, bye_f1))
        else:
            logger.info('Training complete, Training Loss = %f, Dev Joint goal acc = %f, Dev Slot acc = %f'
                        % (tr_loss / global_step, jg_acc, sl_acc))
    else:
        jg_acc = 0.0
        logger.info('Training complete!')

    # Store final model
    try:
        best_score = (best_model['request f1 score'] * model.config.user_request_loss_weight) + \
            (best_model['active domain f1 score'] * model.config.active_domain_loss_weight) + \
            (best_model['goodbye act f1 score'] *
             model.config.user_general_act_loss_weight)
    except AttributeError:
        best_score = 0.0
    best_score += best_model['joint goal accuracy']
    try:
        current_score = (req_f1 * model.config.user_request_loss_weight) + \
                        (dom_f1 * model.config.active_domain_loss_weight) + \
                        (bye_f1 * model.config.user_general_act_loss_weight) if req_f1 is not None else 0.0
    except AttributeError:
        current_score = 0.0
    current_score += jg_acc
    if best_model['joint goal accuracy'] < jg_acc and jg_acc > 0.0:
        update = True
    elif current_score > best_score and current_score > 0.0:
        update = True
    elif best_model['train loss'] > (tr_loss / global_step) and best_model['joint goal accuracy'] == 0.0:
        update = True
    else:
        update = False

    if update:
        logger.info('Final model saved.')
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.n_gpu > 1:
            model.module.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        if temp_scheduler is not None:
            torch.save(temp_scheduler.state_dict(), os.path.join(output_dir, 'temp_scheduler.pt'))
        if args.fp16:
            torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
        clear_checkpoints(args.output_dir)
    else:
        logger.info(
            'Final model not saved, since it is not the best performing model.')


# Function for validation
def train_eval(args, model, device, dev_dataloader):
    """Evaluate Model during training!"""
    accuracy_jg = []
    accuracy_sl = []
    accuracy_req = []
    truepos_req, falsepos_req, falseneg_req = [], [], []
    truepos_dom, falsepos_dom, falseneg_dom = [], [], []
    truepos_bye, falsepos_bye, falseneg_bye = [], [], []
    accuracy_dom = []
    accuracy_bye = []
    turns = []
    for batch in dev_dataloader:
        # Perform with no gradients stored
        with torch.no_grad():
            if 'goodbye_belief' in batch:
                labels = {slot: batch['belief-' + slot].to(device) for slot in model.informable_slot_ids
                          if ('belief-' + slot) in batch}
                request_labels = {slot: batch['request_belief-' + slot].to(device) for slot in model.requestable_slot_ids
                                  if ('request_belief-' + slot) in batch} if args.predict_actions else None
                domain_labels = {domain: batch['domain_belief-' + domain].to(device) for domain in model.domain_ids
                                 if ('domain_belief-' + domain) in batch} if args.predict_actions else None
                goodbye_labels = batch['goodbye_belief'].to(
                    device) if args.predict_actions else None
            else:
                labels = {slot: batch['labels-' + slot].to(device) for slot in model.informable_slot_ids
                          if ('labels-' + slot) in batch}
                request_labels = {slot: batch['request-' + slot].to(device) for slot in model.requestable_slot_ids
                                  if ('request-' + slot) in batch} if args.predict_actions else None
                domain_labels = {domain: batch['active-' + domain].to(device) for domain in model.domain_ids
                                 if ('active-' + domain) in batch} if args.predict_actions else None
                goodbye_labels = batch['goodbye'].to(
                    device) if args.predict_actions else None

            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(
                device) if 'token_type_ids' in batch else None
            attention_mask = batch['attention_mask'].to(
                device) if 'attention_mask' in batch else None

            loss, p, p_req, p_dom, p_bye, _, stats = model(input_ids=input_ids,
                                                           token_type_ids=token_type_ids,
                                                           attention_mask=attention_mask,
                                                           inform_labels=labels,
                                                           request_labels=request_labels,
                                                           domain_labels=domain_labels,
                                                           goodbye_labels=goodbye_labels)

        jg_acc = 0.0
        req_acc = 0.0
        req_tp, req_fp, req_fn = 0.0, 0.0, 0.0
        dom_tp, dom_fp, dom_fn = 0.0, 0.0, 0.0
        dom_acc = 0.0
        for slot in model.informable_slot_ids:
            labels = batch['labels-' + slot].to(device)
            p_ = p[slot]

            acc = (p_.argmax(-1) == labels).reshape(-1).float()
            jg_acc += acc

        if model.config.predict_actions:
            for slot in model.requestable_slot_ids:
                p_req_ = p_req[slot]
                request_labels = batch['request-' + slot].to(device)

                acc = (p_req_.round().int() == request_labels).reshape(-1).float()
                tp = (p_req_.round().int() * (request_labels == 1)).reshape(-1).float()
                fp = (p_req_.round().int() * (request_labels == 0)).reshape(-1).float()
                fn = ((1 - p_req_.round().int()) * (request_labels == 1)).reshape(-1).float()
                req_acc += acc
                req_tp += tp
                req_fp += fp
                req_fn += fn

            domains = [domain for domain in model.domain_ids if f'active-{domain}' in batch]
            for domain in domains:
                p_dom_ = p_dom[domain]
                domain_labels = batch['active-' + domain].to(device)

                acc = (p_dom_.round().int() == domain_labels).reshape(-1).float()
                tp = (p_dom_.round().int() * (domain_labels == 1)).reshape(-1).float()
                fp = (p_dom_.round().int() * (domain_labels == 0)).reshape(-1).float()
                fn = ((1 - p_dom_.round().int()) * (domain_labels == 1)).reshape(-1).float()
                dom_acc += acc
                dom_tp += tp
                dom_fp += fp
                dom_fn += fn

            goodbye_labels = batch['goodbye'].to(device)
            bye_acc = (p_bye.argmax(-1) == goodbye_labels).reshape(-1).float().sum()
            bye_tp = ((p_bye.argmax(-1) > 0) * (goodbye_labels > 0)).reshape(-1).float().sum()
            bye_fp = ((p_bye.argmax(-1) > 0) * (goodbye_labels == 0)).reshape(-1).float().sum()
            bye_fn = ((p_bye.argmax(-1) == 0) * (goodbye_labels > 0)).reshape(-1).float().sum()
        else:
            req_acc, dom_acc, bye_acc = None, None, torch.tensor(0.0)
            req_tp, req_fp, req_fn = None, None, None
            dom_tp, dom_fp, dom_fn = None, None, None
            bye_tp, bye_fp, bye_fn = torch.tensor(
                0.0), torch.tensor(0.0), torch.tensor(0.0)

        sl_acc = sum(jg_acc / len(model.informable_slot_ids)).float()
        jg_acc = sum((jg_acc == len(model.informable_slot_ids)).int()).float()
        req_acc = sum(req_acc / len(model.requestable_slot_ids)).float() if req_acc is not None else torch.tensor(0.0)
        req_tp = sum(req_tp / len(model.requestable_slot_ids)).float() if req_tp is not None else torch.tensor(0.0)
        req_fp = sum(req_fp / len(model.requestable_slot_ids)).float() if req_fp is not None else torch.tensor(0.0)
        req_fn = sum(req_fn / len(model.requestable_slot_ids)).float() if req_fn is not None else torch.tensor(0.0)
        dom_tp = sum(dom_tp / len(model.domain_ids)).float() if dom_tp is not None else torch.tensor(0.0)
        dom_fp = sum(dom_fp / len(model.domain_ids)).float() if dom_fp is not None else torch.tensor(0.0)
        dom_fn = sum(dom_fn / len(model.domain_ids)).float() if dom_fn is not None else torch.tensor(0.0)
        dom_acc = sum(dom_acc / len(model.domain_ids)).float() if dom_acc is not None else torch.tensor(0.0)
        n_turns = (labels >= 0).reshape(-1).sum().float().item()

        accuracy_jg.append(jg_acc.item())
        accuracy_sl.append(sl_acc.item())
        accuracy_req.append(req_acc.item())
        truepos_req.append(req_tp.item())
        falsepos_req.append(req_fp.item())
        falseneg_req.append(req_fn.item())
        accuracy_dom.append(dom_acc.item())
        truepos_dom.append(dom_tp.item())
        falsepos_dom.append(dom_fp.item())
        falseneg_dom.append(dom_fn.item())
        accuracy_bye.append(bye_acc.item())
        truepos_bye.append(bye_tp.item())
        falsepos_bye.append(bye_fp.item())
        falseneg_bye.append(bye_fn.item())
        turns.append(n_turns)

    # Global accuracy reduction across batches
    turns = sum(turns)
    jg_acc = sum(accuracy_jg) / turns
    sl_acc = sum(accuracy_sl) / turns
    if model.config.predict_actions:
        req_acc = sum(accuracy_req) / turns
        req_tp = sum(truepos_req)
        req_fp = sum(falsepos_req)
        req_fn = sum(falseneg_req)
        req_f1 = req_tp + 0.5 * (req_fp + req_fn)
        req_f1 = req_tp / req_f1 if req_f1 != 0.0 else 0.0
        dom_acc = sum(accuracy_dom) / turns
        dom_tp = sum(truepos_dom)
        dom_fp = sum(falsepos_dom)
        dom_fn = sum(falseneg_dom)
        dom_f1 = dom_tp + 0.5 * (dom_fp + dom_fn)
        dom_f1 = dom_tp / dom_f1 if dom_f1 != 0.0 else 0.0
        bye_tp = sum(truepos_bye)
        bye_fp = sum(falsepos_bye)
        bye_fn = sum(falseneg_bye)
        bye_f1 = bye_tp + 0.5 * (bye_fp + bye_fn)
        bye_f1 = bye_tp / bye_f1 if bye_f1 != 0.0 else 0.0
        bye_acc = sum(accuracy_bye) / turns
    else:
        req_acc, dom_acc, bye_acc = None, None, None
        req_f1, dom_f1, bye_f1 = None, None, None

    return jg_acc, sl_acc, req_f1, dom_f1, bye_f1, loss, stats


def evaluate(args, model, device, dataloader, return_eval_output=False):
    """Evaluate Model!"""
    # Evaluate!
    logger.info("***** Running evaluation *****")
    logger.info("  Num Batches = %d", len(dataloader))

    tr_loss = 0.0
    model.eval()
    if return_eval_output:
        ontology = dataloader.dataset.ontology

    # logits = {slot: [] for slot in model.informable_slot_ids}
    accuracy_jg = []
    accuracy_sl = []
    accuracy_req = []
    truepos_req, falsepos_req, falseneg_req = [], [], []
    truepos_dom, falsepos_dom, falseneg_dom = [], [], []
    truepos_bye, falsepos_bye, falseneg_bye = [], [], []
    accuracy_dom = []
    accuracy_bye = []
    turns = []
    if return_eval_output:
        evaluation_output = []
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for batch in epoch_iterator:
        with torch.no_grad():
            if 'goodbye_belief' in batch:
                labels = {slot: batch['belief-' + slot].to(device) for slot in model.informable_slot_ids
                          if ('belief-' + slot) in batch}
                request_labels = {slot: batch['request_belief-' + slot].to(device) for slot in model.requestable_slot_ids
                                  if ('request_belief-' + slot) in batch} if args.predict_actions else None
                domain_labels = {domain: batch['domain_belief-' + domain].to(device) for domain in model.domain_ids
                                 if ('domain_belief-' + domain) in batch} if args.predict_actions else None
                goodbye_labels = batch['goodbye_belief'].to(device) if args.predict_actions else None
            else:
                labels = {slot: batch['labels-' + slot].to(device) for slot in model.informable_slot_ids
                          if ('labels-' + slot) in batch}
                request_labels = {slot: batch['request-' + slot].to(device) for slot in model.requestable_slot_ids
                                  if ('request-' + slot) in batch} if args.predict_actions else None
                domain_labels = {domain: batch['active-' + domain].to(device) for domain in model.domain_ids
                                 if ('active-' + domain) in batch} if args.predict_actions else None
                goodbye_labels = batch['goodbye'].to(device) if args.predict_actions else None

            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

            loss, p, p_req, p_dom, p_bye, _, _ = model(input_ids=input_ids,
                                                       token_type_ids=token_type_ids,
                                                       attention_mask=attention_mask,
                                                       inform_labels=labels,
                                                       request_labels=request_labels,
                                                       domain_labels=domain_labels,
                                                       goodbye_labels=goodbye_labels)

        jg_acc = 0.0
        req_acc = 0.0
        req_tp, req_fp, req_fn = 0.0, 0.0, 0.0
        dom_tp, dom_fp, dom_fn = 0.0, 0.0, 0.0
        dom_acc = 0.0

        if return_eval_output:
            eval_output_batch = []
            for dial_id, dial in enumerate(input_ids):
                for turn_id, turn in enumerate(dial):
                    if turn.sum() != 0:
                        eval_output_batch.append({'dial_idx': dial_id,
                                                  'utt_idx': turn_id,
                                                  'state': {domain: {slot: '' for slot in substate}
                                                            for domain, substate in ontology.items()},
                                                  'predictions': {'state': {domain: {slot: '' for slot in substate}
                                                                            for domain, substate in ontology.items()}}
                                                  })

        for slot in model.informable_slot_ids:
            p_ = p[slot]
            labels = batch['labels-' + slot].to(device)

            if return_eval_output:
                prediction = p_.argmax(-1)

                for sample in eval_output_batch:
                    dom, slt = slot.split('-', 1)
                    pred = prediction[sample['dial_idx']][sample['utt_idx']].item()
                    pred = ontology[dom][slt]['possible_values'][pred]
                    lab = labels[sample['dial_idx']][sample['utt_idx']].item()
                    lab = ontology[dom][slt]['possible_values'][lab]

                    sample['state'][dom][slt] = lab if lab != 'none' else ''
                    sample['predictions']['state'][dom][slt] = pred if pred != 'none' else ''

            if args.temp_scaling > 0.0:
                p_ = torch.log(p_ + 1e-10) / args.temp_scaling
                p_ = torch.softmax(p_, -1)
            else:
                p_ = torch.log(p_ + 1e-10) / 1.0
                p_ = torch.softmax(p_, -1)

            acc = (p_.argmax(-1) == labels).reshape(-1).float()

            jg_acc += acc

        if return_eval_output:
            evaluation_output += deepcopy(eval_output_batch)
            eval_output_batch = []

        if model.config.predict_actions:
            for slot in model.requestable_slot_ids:
                p_req_ = p_req[slot]
                request_labels = batch['request-' + slot].to(device)

                acc = (p_req_.round().int() == request_labels).reshape(-1).float()
                tp = (p_req_.round().int() * (request_labels == 1)).reshape(-1).float()
                fp = (p_req_.round().int() * (request_labels == 0)).reshape(-1).float()
                fn = ((1 - p_req_.round().int()) * (request_labels == 1)).reshape(-1).float()
                req_acc += acc
                req_tp += tp
                req_fp += fp
                req_fn += fn

            domains = [domain for domain in model.domain_ids if f'active-{domain}' in batch]
            for domain in domains:
                p_dom_ = p_dom[domain]
                domain_labels = batch['active-' + domain].to(device)

                acc = (p_dom_.round().int() == domain_labels).reshape(-1).float()
                tp = (p_dom_.round().int() * (domain_labels == 1)).reshape(-1).float()
                fp = (p_dom_.round().int() * (domain_labels == 0)).reshape(-1).float()
                fn = ((1 - p_dom_.round().int()) * (domain_labels == 1)).reshape(-1).float()
                dom_acc += acc
                dom_tp += tp
                dom_fp += fp
                dom_fn += fn

            goodbye_labels = batch['goodbye'].to(device)
            bye_acc = (p_bye.argmax(-1) == goodbye_labels).reshape(-1).float().sum()
            bye_tp = ((p_bye.argmax(-1) > 0) * (goodbye_labels > 0)).reshape(-1).float().sum()
            bye_fp = ((p_bye.argmax(-1) > 0) * (goodbye_labels == 0)).reshape(-1).float().sum()
            bye_fn = ((p_bye.argmax(-1) == 0) * (goodbye_labels > 0)).reshape(-1).float().sum()
        else:
            req_acc, dom_acc, bye_acc = None, None, torch.tensor(0.0)
            req_tp, req_fp, req_fn = None, None, None
            dom_tp, dom_fp, dom_fn = None, None, None
            bye_tp, bye_fp, bye_fn = torch.tensor(
                0.0), torch.tensor(0.0), torch.tensor(0.0)

        sl_acc = sum(jg_acc / len(model.informable_slot_ids)).float()
        jg_acc = sum((jg_acc == len(model.informable_slot_ids)).int()).float()
        req_acc = sum(req_acc / len(model.requestable_slot_ids)).float() if req_acc is not None else torch.tensor(0.0)
        req_tp = sum(req_tp / len(model.requestable_slot_ids)).float() if req_tp is not None else torch.tensor(0.0)
        req_fp = sum(req_fp / len(model.requestable_slot_ids)).float() if req_fp is not None else torch.tensor(0.0)
        req_fn = sum(req_fn / len(model.requestable_slot_ids)).float() if req_fn is not None else torch.tensor(0.0)
        dom_tp = sum(dom_tp / len(model.domain_ids)).float() if dom_tp is not None else torch.tensor(0.0)
        dom_fp = sum(dom_fp / len(model.domain_ids)).float() if dom_fp is not None else torch.tensor(0.0)
        dom_fn = sum(dom_fn / len(model.domain_ids)).float() if dom_fn is not None else torch.tensor(0.0)
        dom_acc = sum(dom_acc / len(model.domain_ids)).float() if dom_acc is not None else torch.tensor(0.0)
        n_turns = (labels >= 0).reshape(-1).sum().float().item()

        accuracy_jg.append(jg_acc.item())
        accuracy_sl.append(sl_acc.item())
        accuracy_req.append(req_acc.item())
        truepos_req.append(req_tp.item())
        falsepos_req.append(req_fp.item())
        falseneg_req.append(req_fn.item())
        accuracy_dom.append(dom_acc.item())
        truepos_dom.append(dom_tp.item())
        falsepos_dom.append(dom_fp.item())
        falseneg_dom.append(dom_fn.item())
        accuracy_bye.append(bye_acc.item())
        truepos_bye.append(bye_tp.item())
        falsepos_bye.append(bye_fp.item())
        falseneg_bye.append(bye_fn.item())
        turns.append(n_turns)
        tr_loss += loss.item()

    # for slot in logits:
    #     logits[slot] = torch.cat(logits[slot], 0)

    # Global accuracy reduction across batches
    turns = sum(turns)
    jg_acc = sum(accuracy_jg) / turns
    sl_acc = sum(accuracy_sl) / turns
    if model.config.predict_actions:
        req_acc = sum(accuracy_req) / turns
        req_tp = sum(truepos_req)
        req_fp = sum(falsepos_req)
        req_fn = sum(falseneg_req)
        req_f1 = req_tp + 0.5 * (req_fp + req_fn)
        req_f1 = req_tp / req_f1 if req_f1 != 0.0 else 0.0
        dom_acc = sum(accuracy_dom) / turns
        dom_tp = sum(truepos_dom)
        dom_fp = sum(falsepos_dom)
        dom_fn = sum(falseneg_dom)
        dom_f1 = dom_tp + 0.5 * (dom_fp + dom_fn)
        dom_f1 = dom_tp / dom_f1 if dom_f1 != 0.0 else 0.0
        bye_tp = sum(truepos_bye)
        bye_fp = sum(falsepos_bye)
        bye_fn = sum(falseneg_bye)
        bye_f1 = bye_tp + 0.5 * (bye_fp + bye_fn)
        bye_f1 = bye_tp / bye_f1 if bye_f1 != 0.0 else 0.0
        bye_acc = sum(accuracy_bye) / turns
    else:
        req_acc, dom_acc, bye_acc = None, None, None
        req_f1, dom_f1, bye_f1 = None, None, None

    if return_eval_output:
        dial_idx = 0
        for sample in evaluation_output:
            if dial_idx == 0 and sample['dial_idx'] == 0 and sample['utt_idx'] == 0:
                dial_idx = 0
            elif dial_idx == 0 and sample['dial_idx'] != 0 and sample['utt_idx'] == 0:
                dial_idx += 1
            elif sample['utt_idx'] == 0:
                dial_idx += 1
            sample['dial_idx'] = dial_idx

        return jg_acc, sl_acc, req_f1, dom_f1, bye_f1, tr_loss / len(dataloader), evaluation_output
    return jg_acc, sl_acc, req_f1, dom_f1, bye_f1, tr_loss / len(dataloader)
