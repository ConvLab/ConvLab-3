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
"""Training and evaluation utils"""

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
    try:
        informable_slot_ids = model.setsumbt.informable_slot_ids
    except:
        informable_slot_ids = model.informable_slot_ids
    for slot in informable_slot_ids:
        model.add_value_candidates(slot, values[slot], replace=True)


def log_info(global_step, loss, jg_acc=None, sl_acc=None, req_f1=None, dom_f1=None, gen_f1=None, stats=None):
    """
    Log training statistics.

    Args:
        global_step: Number of global training steps completed
        loss: Training loss
        jg_acc: Joint goal accuracy
        sl_acc: Slot accuracy
        req_f1: Request prediction F1 score
        dom_f1: Active domain prediction F1 score
        gen_f1: General action prediction F1 score
        stats: Uncertainty measure statistics of model
    """
    if type(global_step) == int:
        info = f"{global_step} steps complete, "
        info += f"Loss since last update: {loss}. Validation set stats: "
    elif global_step == 'training_complete':
        info = f"Training Complete, "
        info += f"Validation set stats: "
    elif global_step == 'dev':
        info = f"Validation set stats: Loss: {loss}, "
    elif global_step == 'test':
        info = f"Test set stats: Loss: {loss}, "
    info += f"Joint Goal Acc: {jg_acc}, Slot Acc: {sl_acc}, "
    if req_f1 is not None:
        info += f"Request F1 Score: {req_f1}, Active Domain F1 Score: {dom_f1}, "
        info += f"General Action F1 Score: {gen_f1}"
    logger.info(info)

    if type(global_step) == int:
        tb_writer.add_scalar('JointGoalAccuracy/Dev', jg_acc, global_step)
        tb_writer.add_scalar('SlotAccuracy/Dev', sl_acc, global_step)
        if req_f1 is not None:
            tb_writer.add_scalar('RequestF1Score/Dev', req_f1, global_step)
            tb_writer.add_scalar('ActiveDomainF1Score/Dev', dom_f1, global_step)
            tb_writer.add_scalar('GeneralActionF1Score/Dev', gen_f1, global_step)
        tb_writer.add_scalar('Loss/Dev', loss, global_step)

        if stats:
            for slot, stats_slot in stats.items():
                for key, item in stats_slot.items():
                    tb_writer.add_scalar(f'{key}_{slot}/Dev', item, global_step)


def get_input_dict(batch: dict,
                   predict_actions: bool,
                   model_informable_slot_ids: list,
                   model_requestable_slot_ids: list = None,
                   model_domain_ids: list = None,
                   device = 'cpu') -> dict:
    """
    Produce model input arguments

    Args:
        batch: Batch of data from the dataloader
        predict_actions: Model should predict user actions if set true
        model_informable_slot_ids: List of model dialogue state slots
        model_requestable_slot_ids: List of model requestable slots
        model_domain_ids: List of model domains
        device: Current torch device in use

    Returns:
        input_dict: Dictrionary containing model inputs for the batch
    """
    input_dict = dict()

    input_dict['input_ids'] = batch['input_ids'].to(device)
    input_dict['token_type_ids'] = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
    input_dict['attention_mask'] = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

    if any('belief_state' in key for key in batch):
        input_dict['state_labels'] = {slot: batch['belief_state-' + slot].to(device)
                                      for slot in model_informable_slot_ids
                                      if ('belief_state-' + slot) in batch}
        if predict_actions:
            input_dict['request_labels'] = {slot: batch['request_probs-' + slot].to(device)
                                            for slot in model_requestable_slot_ids
                                            if ('request_probs-' + slot) in batch}
            input_dict['active_domain_labels'] = {domain: batch['active_domain_probs-' + domain].to(device)
                                                  for domain in model_domain_ids
                                                  if ('active_domain_probs-' + domain) in batch}
            input_dict['general_act_labels'] = batch['general_act_probs'].to(device)
    else:
        input_dict['state_labels'] = {slot: batch['state_labels-' + slot].to(device)
                                      for slot in model_informable_slot_ids if ('state_labels-' + slot) in batch}
        if predict_actions:
            input_dict['request_labels'] = {slot: batch['request_labels-' + slot].to(device)
                                            for slot in model_requestable_slot_ids
                                            if ('request_labels-' + slot) in batch}
            input_dict['active_domain_labels'] = {domain: batch['active_domain_labels-' + domain].to(device)
                                                  for domain in model_domain_ids
                                                  if ('active_domain_labels-' + domain) in batch}
            input_dict['general_act_labels'] = batch['general_act_labels'].to(device)

    return input_dict


def train(args, model, device, train_dataloader, dev_dataloader, slots: dict, slots_dev: dict):
    """
    Train the SetSUMBT model.

    Args:
        args: Runtime arguments
        model: SetSUMBT Model instance to train
        device: Torch device to use during training
        train_dataloader: Dataloader containing the training data
        dev_dataloader: Dataloader containing the validation set data
        slots: Model ontology used for training
        slots_dev: Model ontology used for evaluating on the validation set
    """

    # Calculate the total number of training steps to be performed
    if args.max_training_steps > 0:
        t_total = args.max_training_steps
        args.num_train_epochs = (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        args.num_train_epochs = args.max_training_steps // args.num_train_epochs
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
            "lr": args.learning_rate
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
                  'general act f1 score': 0.0,
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

            jg_acc, sl_acc, req_f1, dom_f1, gen_f1, _, _ = evaluate(args, model, device, dev_dataloader, is_train=True)

            # Set model back to training mode
            model.train()
            model.zero_grad()
            set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots, load_slots=False)
        else:
            jg_acc, req_f1, dom_f1, gen_f1 = 0.0, 0.0, 0.0, 0.0

        best_model['joint goal accuracy'] = jg_acc
        best_model['request f1 score'] = req_f1
        best_model['active domain f1 score'] = dom_f1
        best_model['general act f1 score'] = gen_f1

    # Log training set up
    logger.info(f"Device: {device}, Number of GPUs: {args.n_gpu}, FP16 training: {args.fp16}")
    logger.info("***** Running training *****")
    logger.info(f"  Num Batches = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

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
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
        except ValueError:
            logger.info(f"  Starting fine-tuning.")

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
            input_dict = get_input_dict(batch, args.predict_actions, model.setsumbt.informable_slot_ids,
                                        model.setsumbt.requestable_slot_ids, model.setsumbt.domain_ids, device)

            # Set up temperature scaling for the model
            if temp_scheduler is not None:
                model.setsumbt.temp = temp_scheduler.temp()

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
                global_step += 1

            # Save model checkpoint
            if global_step % args.save_steps == 0:
                logging_loss = tr_loss - logging_loss

                # Evaluate model
                if args.do_eval:
                    # Set up model for evaluation
                    model.eval()
                    set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots_dev, load_slots=False)

                    jg_acc, sl_acc, req_f1, dom_f1, gen_f1, loss, stats = evaluate(args, model, device, dev_dataloader,
                                                                                   is_train=True)
                    # Log model eval information
                    log_info(global_step, logging_loss / args.save_steps, jg_acc, sl_acc, req_f1, dom_f1, gen_f1, stats)

                    # Set model back to training mode
                    model.train()
                    model.zero_grad()
                    set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots, load_slots=False)
                else:
                    log_info(global_step, logging_loss / args.save_steps)

                logging_loss = tr_loss

                # Compute the score of the best model
                try:
                    best_score = best_model['request f1 score'] * model.config.user_request_loss_weight
                    best_score += best_model['active domain f1 score'] * model.config.active_domain_loss_weight
                    best_score += best_model['general act f1 score'] * model.config.user_general_act_loss_weight
                except AttributeError:
                    best_score = 0.0
                best_score += best_model['joint goal accuracy']

                # Compute the score of the current model
                try:
                    current_score = req_f1 * model.config.user_request_loss_weight if req_f1 is not None else 0.0
                    current_score += dom_f1 * model.config.active_domain_loss_weight if dom_f1 is not None else 0.0
                    current_score += gen_f1 * model.config.user_general_act_loss_weight if gen_f1 is not None else 0.0
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
                        best_model['general act f1 score'] = gen_f1
                    best_model['train loss'] = tr_loss / global_step

                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
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

        steps_trained_in_current_epoch = 0
        logger.info(f'Epoch {e + 1} complete, average training loss = {tr_loss / global_step}')

        if args.max_training_steps > 0 and global_step > args.max_training_steps:
            train_iterator.close()
            break
        if args.patience > 0 and steps_since_last_update >= args.patience:
            train_iterator.close()
            logger.info(f'Model has not improved for at least {args.patience} steps. Training stopped!')
            break

    # Evaluate final model
    if args.do_eval:
        model.eval()
        set_ontology_embeddings(model.module if args.n_gpu > 1 else model, slots_dev, load_slots=False)

        jg_acc, sl_acc, req_f1, dom_f1, gen_f1, loss, stats = evaluate(args, model, device, dev_dataloader,
                                                                       is_train=True)

        log_info('training_complete', tr_loss / global_step, jg_acc, sl_acc, req_f1, dom_f1, gen_f1)
    else:
        logger.info('Training complete!')

    # Store final model
    try:
        best_score = best_model['request f1 score'] * model.config.user_request_loss_weight
        best_score += best_model['active domain f1 score'] * model.config.active_domain_loss_weight
        best_score += best_model['general act f1 score'] * model.config.user_general_act_loss_weight
    except AttributeError:
        best_score = 0.0
    best_score += best_model['joint goal accuracy']
    try:
        current_score = req_f1 * model.config.user_request_loss_weight if req_f1 is not None else 0.0
        current_score += dom_f1 * model.config.active_domain_loss_weight if dom_f1 is not None else 0.0
        current_score += gen_f1 * model.config.user_general_act_loss_weight if gen_f1 is not None else 0.0
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
        logger.info('Final model not saved, as it is not the best performing model.')


def evaluate(args, model, device, dataloader, return_eval_output=False, is_train=False):
    """
    Evaluate model

    Args:
        args: Runtime arguments
        model: SetSUMBT model instance
        device: Torch device in use
        dataloader: Dataloader of data to evaluate on
        return_eval_output: If true return predicted and true states for all dialogues evaluated in semantic format
        is_train: If true model is training and no logging is performed

    Returns:
        out: Evaluated model statistics
    """
    return_eval_output = False if is_train else return_eval_output
    if not is_train:
        logger.info("***** Running evaluation *****")
        logger.info("  Num Batches = %d", len(dataloader))

    tr_loss = 0.0
    model.eval()
    if return_eval_output:
        ontology = dataloader.dataset.ontology

    accuracy_jg = []
    accuracy_sl = []
    truepos_req, falsepos_req, falseneg_req = [], [], []
    truepos_dom, falsepos_dom, falseneg_dom = [], [], []
    truepos_gen, falsepos_gen, falseneg_gen = [], [], []
    turns = []
    if return_eval_output:
        evaluation_output = []
    epoch_iterator = tqdm(dataloader, desc="Iteration") if not is_train else dataloader
    for batch in epoch_iterator:
        with torch.no_grad():
            input_dict = get_input_dict(batch, args.predict_actions, model.setsumbt.informable_slot_ids,
                                        model.setsumbt.requestable_slot_ids, model.setsumbt.domain_ids, device)

            loss, p, p_req, p_dom, p_gen, _, stats = model(**input_dict)

        jg_acc = 0.0
        num_inform_slots = 0.0
        req_acc = 0.0
        req_tp, req_fp, req_fn = 0.0, 0.0, 0.0
        dom_tp, dom_fp, dom_fn = 0.0, 0.0, 0.0
        dom_acc = 0.0

        if return_eval_output:
            eval_output_batch = []
            for dial_id, dial in enumerate(input_dict['input_ids']):
                for turn_id, turn in enumerate(dial):
                    if turn.sum() != 0:
                        eval_output_batch.append({'dial_idx': dial_id,
                                                  'utt_idx': turn_id,
                                                  'state': dict(),
                                                  'predictions': {'state': dict()}
                                                  })

        for slot in model.setsumbt.informable_slot_ids:
            p_ = p[slot]
            state_labels = batch['state_labels-' + slot].to(device)

            if return_eval_output:
                prediction = p_.argmax(-1)

                for sample in eval_output_batch:
                    dom, slt = slot.split('-', 1)
                    lab = state_labels[sample['dial_idx']][sample['utt_idx']].item()
                    lab = ontology[dom][slt]['possible_values'][lab] if lab != -1 else 'NOT_IN_ONTOLOGY'
                    pred = prediction[sample['dial_idx']][sample['utt_idx']].item()
                    pred = ontology[dom][slt]['possible_values'][pred]

                    if dom not in sample['state']:
                        sample['state'][dom] = dict()
                        sample['predictions']['state'][dom] = dict()

                    sample['state'][dom][slt] = lab if lab != 'none' else ''
                    sample['predictions']['state'][dom][slt] = pred if pred != 'none' else ''

            if args.temp_scaling > 0.0:
                p_ = torch.log(p_ + 1e-10) / args.temp_scaling
                p_ = torch.softmax(p_, -1)
            else:
                p_ = torch.log(p_ + 1e-10) / 1.0
                p_ = torch.softmax(p_, -1)

            acc = (p_.argmax(-1) == state_labels).reshape(-1).float()

            jg_acc += acc
            num_inform_slots += (state_labels != -1).float().reshape(-1)

        if return_eval_output:
            for sample in eval_output_batch:
                sample['dial_idx'] = batch['dialogue_ids'][sample['utt_idx']][sample['dial_idx']]
                evaluation_output.append(deepcopy(sample))
            eval_output_batch = []

        if model.config.predict_actions:
            for slot in model.setsumbt.requestable_slot_ids:
                p_req_ = p_req[slot]
                request_labels = batch['request_labels-' + slot].to(device)

                acc = (p_req_.round().int() == request_labels).reshape(-1).float()
                tp = (p_req_.round().int() * (request_labels == 1)).reshape(-1).float()
                fp = (p_req_.round().int() * (request_labels == 0)).reshape(-1).float()
                fn = ((1 - p_req_.round().int()) * (request_labels == 1)).reshape(-1).float()
                req_acc += acc
                req_tp += tp
                req_fp += fp
                req_fn += fn

            domains = [domain for domain in model.setsumbt.domain_ids if f'active_domain_labels-{domain}' in batch]
            for domain in domains:
                p_dom_ = p_dom[domain]
                active_domain_labels = batch['active_domain_labels-' + domain].to(device)

                acc = (p_dom_.round().int() == active_domain_labels).reshape(-1).float()
                tp = (p_dom_.round().int() * (active_domain_labels == 1)).reshape(-1).float()
                fp = (p_dom_.round().int() * (active_domain_labels == 0)).reshape(-1).float()
                fn = ((1 - p_dom_.round().int()) * (active_domain_labels == 1)).reshape(-1).float()
                dom_acc += acc
                dom_tp += tp
                dom_fp += fp
                dom_fn += fn

            general_act_labels = batch['general_act_labels'].to(device)
            gen_tp = ((p_gen.argmax(-1) > 0) * (general_act_labels > 0)).reshape(-1).float().sum()
            gen_fp = ((p_gen.argmax(-1) > 0) * (general_act_labels == 0)).reshape(-1).float().sum()
            gen_fn = ((p_gen.argmax(-1) == 0) * (general_act_labels > 0)).reshape(-1).float().sum()
        else:
            req_tp, req_fp, req_fn = None, None, None
            dom_tp, dom_fp, dom_fn = None, None, None
            gen_tp, gen_fp, gen_fn = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        jg_acc = jg_acc[num_inform_slots > 0]
        num_inform_slots = num_inform_slots[num_inform_slots > 0]
        sl_acc = sum(jg_acc / num_inform_slots).float()
        jg_acc = sum((jg_acc == num_inform_slots).int()).float()
        if req_tp is not None and model.setsumbt.requestable_slot_ids:
            req_tp = sum(req_tp / len(model.setsumbt.requestable_slot_ids)).float()
            req_fp = sum(req_fp / len(model.setsumbt.requestable_slot_ids)).float()
            req_fn = sum(req_fn / len(model.setsumbt.requestable_slot_ids)).float()
        else:
            req_tp, req_fp, req_fn = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        dom_tp = sum(dom_tp / len(model.setsumbt.domain_ids)).float() if dom_tp is not None else torch.tensor(0.0)
        dom_fp = sum(dom_fp / len(model.setsumbt.domain_ids)).float() if dom_fp is not None else torch.tensor(0.0)
        dom_fn = sum(dom_fn / len(model.setsumbt.domain_ids)).float() if dom_fn is not None else torch.tensor(0.0)
        n_turns = num_inform_slots.size(0)

        accuracy_jg.append(jg_acc.item())
        accuracy_sl.append(sl_acc.item())
        truepos_req.append(req_tp.item())
        falsepos_req.append(req_fp.item())
        falseneg_req.append(req_fn.item())
        truepos_dom.append(dom_tp.item())
        falsepos_dom.append(dom_fp.item())
        falseneg_dom.append(dom_fn.item())
        truepos_gen.append(gen_tp.item())
        falsepos_gen.append(gen_fp.item())
        falseneg_gen.append(gen_fn.item())
        turns.append(n_turns)
        tr_loss += loss.item()

    # Global accuracy reduction across batches
    turns = sum(turns)
    jg_acc = sum(accuracy_jg) / turns
    sl_acc = sum(accuracy_sl) / turns
    if model.config.predict_actions:
        req_tp = sum(truepos_req)
        req_fp = sum(falsepos_req)
        req_fn = sum(falseneg_req)
        req_f1 = req_tp + 0.5 * (req_fp + req_fn)
        req_f1 = req_tp / req_f1 if req_f1 != 0.0 else 0.0
        dom_tp = sum(truepos_dom)
        dom_fp = sum(falsepos_dom)
        dom_fn = sum(falseneg_dom)
        dom_f1 = dom_tp + 0.5 * (dom_fp + dom_fn)
        dom_f1 = dom_tp / dom_f1 if dom_f1 != 0.0 else 0.0
        gen_tp = sum(truepos_gen)
        gen_fp = sum(falsepos_gen)
        gen_fn = sum(falseneg_gen)
        gen_f1 = gen_tp + 0.5 * (gen_fp + gen_fn)
        gen_f1 = gen_tp / gen_f1 if gen_f1 != 0.0 else 0.0
    else:
        req_f1, dom_f1, gen_f1 = None, None, None

    if return_eval_output:
        return jg_acc, sl_acc, req_f1, dom_f1, gen_f1, tr_loss / len(dataloader), evaluation_output
    if is_train:
        return jg_acc, sl_acc, req_f1, dom_f1, gen_f1, tr_loss / len(dataloader), stats
    return jg_acc, sl_acc, req_f1, dom_f1, gen_f1, tr_loss / len(dataloader)
