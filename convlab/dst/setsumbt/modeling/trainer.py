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
"""SetSUMBT Trainer Class"""

import random
import os
from copy import deepcopy
import pdb

import torch
from torch.nn import DataParallel
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
try:
    from apex import amp
except ModuleNotFoundError:
    print('Apex not used')

from convlab.dst.setsumbt.utils import clear_checkpoints
from convlab.dst.setsumbt.datasets import JointGoalAccuracy, BeliefStateUncertainty, ActPredictionAccuracy, Metrics
from convlab.dst.setsumbt.modeling import LinearTemperatureScheduler
from convlab.dst.setsumbt.utils import EnsembleAggregator


class SetSUMBTTrainer:
    """Trainer class for SetSUMBT Model"""

    def __init__(self,
                 args,
                 model,
                 tokenizer,
                 train_dataloader,
                 validation_dataloader,
                 logger,
                 tb_writer,
                 device='cpu'):
        """
        Initialise the trainer class.

        Args:
            args (argparse.Namespace): Arguments passed to the script
            model (torch.nn.Module): SetSUMBT to be trained/evaluated
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to encode the data
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training data
            validation_dataloader (torch.utils.data.DataLoader): Dataloader for validation data
            logger (logging.Logger): Logger to log training progress
            tb_writer (tensorboardX.SummaryWriter): Tensorboard writer to log training progress
            device (str): Device to use for training
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.logger = logger
        self.tb_writer = tb_writer
        self.device = device

        # Initialise metrics
        if self.validation_dataloader is not None:
            self.joint_goal_accuracy = JointGoalAccuracy(self.args.dataset, validation_dataloader.dataset.set_type)
            self.belief_state_uncertainty_metrics = BeliefStateUncertainty()
            self.ensemble_aggregator = EnsembleAggregator()
            if self.args.predict_actions:
                self.request_accuracy = ActPredictionAccuracy('request', binary=True)
                self.active_domain_accuracy = ActPredictionAccuracy('active_domain', binary=True)
                self.general_act_accuracy = ActPredictionAccuracy('general_act', binary=False)

        self._set_seed()

        if train_dataloader is not None:
            self.training_mode(load_slots=True)
            self._configure_optimiser()
            self._configure_schedulers()

            # Set up fp16 and multi gpu usage
            if self.args.fp16:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                            opt_level=self.args.fp16_opt_level)
            if self.args.n_gpu > 1:
                self.model = DataParallel(self.model)

        # Initialise training parameters
        self.best_model = Metrics(joint_goal_accuracy=0.0,
                                  training_loss=np.inf)
        self.global_step = 0
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0

        logger.info(f"Device: {device}, Number of GPUs: {args.n_gpu}, FP16 training: {args.fp16}")

    def _configure_optimiser(self):
        """Configure the optimiser for training."""
        assert self.train_dataloader is not None
        # Group weight decay and no decay parameters in the model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                           and 'value_embeddings' not in n],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                           and 'value_embeddings' not in n],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            },
        ]

        # Initialise the optimizer
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

    def _configure_schedulers(self):
        """Configure the learning rate and temperature schedulers for training."""
        assert self.train_dataloader is not None
        # Calculate the total number of training steps to be performed
        if self.args.max_training_steps > 0:
            self.args.num_train_epochs = (len(self.train_dataloader) // self.args.gradient_accumulation_steps) + 1
            self.args.num_train_epochs = self.args.max_training_steps // self.args.num_train_epochs
        else:
            self.args.max_training_steps = len(self.train_dataloader) // self.args.gradient_accumulation_steps
            self.args.max_training_steps *= self.args.num_train_epochs

        if self.args.save_steps <= 0:
            self.args.save_steps = len(self.train_dataloader) // self.args.gradient_accumulation_steps

        # Initialise linear lr scheduler
        self.args.num_warmup_steps = int(self.args.max_training_steps * self.args.warmup_proportion)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.args.num_warmup_steps,
                                                            num_training_steps=self.args.max_training_steps)

        # Initialise distillation temp scheduler
        if self.model.config.loss_function in ['distillation']:
            self.temp_scheduler = LinearTemperatureScheduler(total_steps=self.args.max_training_steps,
                                                             base_temp=self.args.annealing_base_temp,
                                                             cycle_len=self.args.annealing_cycle_len)
        else:
            self.temp_scheduler = None

    def _set_seed(self):
        """Set the seed for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)
        self.logger.info('Seed set to %d.' % self.args.seed)

    @staticmethod
    def _set_ontology_embeddings(model, slots, load_slots=True):
        """
        Set the ontology embeddings for the model.

        Args:
            model (torch.nn.Module): Model to set the ontology embeddings for.
            slots (dict): Dictionary of slot names and their corresponding information.
            load_slots (bool): Whether to load/reload the slot embeddings.
        """
        # Get slot and value embeddings
        values = {slot: slots[slot][1] for slot in slots}

        # Load model ontology
        if load_slots:
            slots = {slot: embs for slot, embs in slots.items()}
            model.add_slot_candidates(slots)
        try:
            informable_slot_ids = model.setsumbt.config.informable_slot_ids
        except AttributeError:
            informable_slot_ids = model.config.informable_slot_ids
        for slot in informable_slot_ids:
            model.add_value_candidates(slot, values[slot], replace=True)

    def set_ontology_embeddings(self, slots, load_slots=True):
        """
        Set the ontology embeddings for the model.

        Args:
            slots (dict): Dictionary of slot names and their corresponding information.
            load_slots (bool): Whether to load/reload the slot embeddings.
        """
        self._set_ontology_embeddings(self.model, slots, load_slots=load_slots)

    def load_state(self):
        """Load the model, optimiser and schedulers state from a previous run."""
        if os.path.isfile(os.path.join(self.args.model_name_or_path, 'optimizer.pt')):
            self.logger.info("Optimizer loaded from previous run.")
            self.optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, 'optimizer.pt')))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, 'lr_scheduler.pt')))
            if self.temp_scheduler is not None:
                self.temp_scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path,
                                                                            'temp_scheduler.pt')))
            if self.args.fp16 and os.path.isfile(os.path.join(self.args.model_name_or_path, 'amp.pt')):
                self.logger.info("FP16 Apex Amp loaded from previous run.")
                amp.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, 'amp.pt')))

        # Evaluate initialised model
        if self.args.do_eval:
            self.eval_mode()
            metrics = self.evaluate(is_train=True)
            self.training_mode()

            best_model = metrics
            best_model.training_loss = np.inf

    def save_state(self):
        """Save the model, optimiser and schedulers state for future runs."""
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.tokenizer.save_pretrained(output_dir)
        if self.args.n_gpu > 1:
            self.model.module.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "lr_scheduler.pt"))
        if self.temp_scheduler is not None:
            torch.save(self.temp_scheduler.state_dict(), os.path.join(output_dir, 'temp_scheduler.pt'))
        if self.args.fp16:
            torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))

        # Remove older training checkpoints
        clear_checkpoints(self.args.output_dir, self.args.keep_models)

    def training_mode(self, load_slots=False):
        """
        Set the model and trainer to training mode.

        Args:
            load_slots (bool): Whether to load/reload the slot embeddings.
        """
        assert self.train_dataloader is not None
        self.model.train()
        self.tokenizer.set_setsumbt_ontology(self.train_dataloader.dataset.ontology)
        self.model.zero_grad()
        self.set_ontology_embeddings(self.train_dataloader.dataset.ontology_embeddings, load_slots=load_slots)

    def eval_mode(self, load_slots=False):
        """
        Set the model and trainer to evaluation mode.

        Args:
            load_slots (bool): Whether to load/reload the slot embeddings.
        """
        self.model.eval()
        self.model.zero_grad()
        self.tokenizer.set_setsumbt_ontology(self.validation_dataloader.dataset.ontology)
        self.set_ontology_embeddings(self.validation_dataloader.dataset.ontology_embeddings, load_slots=load_slots)

    def log_info(self, metrics, logging_stage='update'):
        """
        Log information about the training/evaluation.

        Args:
            metrics (Metrics): Metrics object containing the relevant information.
            logging_stage (str): The stage of the training/evaluation to log.
        """
        if logging_stage == "update":
            info = f"{self.global_step} steps complete, "
            info += f"Loss since last update: {metrics.training_loss}."
            self.logger.info(info)
            self.logger.info("Validation set statistics:")
        elif logging_stage == 'training_complete':
            self.logger.info("Training Complete.")
            self.logger.info("Validation set statistics:")
        elif logging_stage == 'dev':
            self.logger.info("Validation set statistics:")
            self.logger.info(f"\tLoss: {metrics.validation_loss}")
        elif logging_stage == 'test':
            self.logger.info("Test set statistics:")
            self.logger.info(f"\tLoss: {metrics.validation_loss}")
        self.logger.info(f"\tJoint Goal Accuracy: {metrics.joint_goal_accuracy}")
        self.logger.info(f"\tGoal Slot F1 Score: {metrics.slot_f1}")
        self.logger.info(f"\tGoal Slot Precision: {metrics.slot_precision}")
        self.logger.info(f"\tGoal Slot Recall: {metrics.slot_recall}")
        self.logger.info(f"\tJoint Goal ECE: {metrics.joint_goal_ece}")
        self.logger.info(f"\tJoint Goal L2-Error: {metrics.joint_l2_error}")
        self.logger.info(f"\tJoint Goal L2-Error Ratio: {metrics.joint_l2_error_ratio}")
        if 'request_f1' in metrics:
            self.logger.info(f"\tRequest Action F1 Score: {metrics.request_f1}")
            self.logger.info(f"\tActive Domain F1 Score: {metrics.active_domain_f1}")
            self.logger.info(f"\tGeneral Action F1 Score: {metrics.general_act_f1}")

        # Log to tensorboard
        if logging_stage == "update":
            self.tb_writer.add_scalar('JointGoalAccuracy/Dev', metrics.joint_goal_accuracy, self.global_step)
            self.tb_writer.add_scalar('SlotAccuracy/Dev', metrics.slot_accuracy, self.global_step)
            self.tb_writer.add_scalar('SlotF1/Dev', metrics.slot_f1, self.global_step)
            self.tb_writer.add_scalar('SlotPrecision/Dev', metrics.slot_precision, self.global_step)
            self.tb_writer.add_scalar('JointGoalECE/Dev', metrics.joint_goal_ece, self.global_step)
            self.tb_writer.add_scalar('JointGoalL2ErrorRatio/Dev', metrics.joint_l2_error_ratio, self.global_step)
            if 'request_f1' in metrics:
                self.tb_writer.add_scalar('RequestF1Score/Dev', metrics.request_f1, self.global_step)
                self.tb_writer.add_scalar('ActiveDomainF1Score/Dev', metrics.active_domain_f1, self.global_step)
                self.tb_writer.add_scalar('GeneralActionF1Score/Dev', metrics.general_act_f1, self.global_step)
            self.tb_writer.add_scalar('Loss/Dev', metrics.validation_loss, self.global_step)

            if 'belief_state_summary' in metrics:
                for slot, stats_slot in metrics.belief_state_summary.items():
                    for key, item in stats_slot.items():
                        self.tb_writer.add_scalar(f'{key}_{slot}/Dev', item, self.global_step)

    def get_input_dict(self, batch: dict) -> dict:
        """
        Get the input dictionary for the model.

        Args:
            batch (dict): The batch of data to be passed to the model.

        Returns:
            input_dict (dict): The input dictionary for the model.
        """
        input_dict = dict()

        # Add the input ids, token type ids, and attention mask
        input_dict['input_ids'] = batch['input_ids'].to(self.device)
        input_dict['token_type_ids'] = batch['token_type_ids'].to(self.device) if 'token_type_ids' in batch else None
        input_dict['attention_mask'] = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None

        # Add the labels
        if any('belief_state' in key for key in batch):
            input_dict['state_labels'] = {slot: batch['belief_state-' + slot].to(self.device)
                                          for slot in self.model.setsumbt.config.informable_slot_ids
                                          if ('belief_state-' + slot) in batch}
            if self.args.predict_actions:
                input_dict['request_labels'] = {slot: batch['request_probabilities-' + slot].to(self.device)
                                                for slot in self.model.setsumbt.config.requestable_slot_ids
                                                if ('request_probabilities-' + slot) in batch}
                input_dict['active_domain_labels'] = {domain: batch['active_domain_probabilities-' + domain].to(self.device)
                                                      for domain in self.model.setsumbt.config.domain_ids
                                                      if ('active_domain_probabilities-' + domain) in batch}
                input_dict['general_act_labels'] = batch['general_act_probabilities'].to(self.device)
        else:
            input_dict['state_labels'] = {slot: batch['state_labels-' + slot].to(self.device)
                                          for slot in self.model.setsumbt.config.informable_slot_ids
                                          if ('state_labels-' + slot) in batch}
            if self.args.predict_actions:
                input_dict['request_labels'] = {slot: batch['request_labels-' + slot].to(self.device)
                                                for slot in self.model.setsumbt.config.requestable_slot_ids
                                                if ('request_labels-' + slot) in batch}
                input_dict['active_domain_labels'] = {domain: batch['active_domain_labels-' + domain].to(self.device)
                                                      for domain in self.model.setsumbt.config.domain_ids
                                                      if ('active_domain_labels-' + domain) in batch}
                input_dict['general_act_labels'] = batch['general_act_labels'].to(self.device)

        return input_dict

    def train(self):
        """Train the SetSUMBT model."""
        # Set the model to training mode
        self.training_mode(load_slots=True)
        self.load_state()

        # Log training set up
        self.logger.info("***** Running training *****")
        self.logger.info(f"\tNum Batches = {len(self.train_dataloader)}")
        self.logger.info(f"\tNum Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"\tGradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"\tTotal optimization steps = {self.args.max_training_steps}")

        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.args.model_name_or_path.split("-")[-1].split("/")[0]
                self.global_step = int(checkpoint_suffix)
                self.epochs_trained = len(self.train_dataloader) // self.args.gradient_accumulation_steps
                self.steps_trained_in_current_epoch = self.global_step % self.epochs_trained
                self.epochs_trained = self.global_step // self.epochs_trained

                self.logger.info("\tContinuing training from checkpoint, will skip to saved global_step")
                self.logger.info(f"\tContinuing training from epoch {self.epochs_trained}")
                self.logger.info(f"\tContinuing training from global step {self.global_step}")
                self.logger.info(f"\tWill skip the first {self.steps_trained_in_current_epoch} steps in the first epoch")
            except ValueError:
                self.logger.info(f"\tStarting fine-tuning.")

        # Prepare iterator for training
        tr_loss, logging_loss = 0.0, 0.0
        train_iterator = trange(self.epochs_trained, int(self.args.num_train_epochs), desc="Epoch")

        steps_since_last_update = 0
        # Perform training
        for e in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            # Iterate over all batches
            for step, batch in enumerate(epoch_iterator):
                # Skip batches already trained on
                if step < self.steps_trained_in_current_epoch:
                    continue

                # Extract all label dictionaries from the batch
                input_dict = self.get_input_dict(batch)

                # Set up temperature scaling for the model
                if self.temp_scheduler is not None:
                    self.model.setsumbt.temp = self.temp_scheduler.temp()

                # Forward pass to obtain loss
                output = self.model(**input_dict)

                if self.args.n_gpu > 1:
                    output.loss = output.loss.mean()

                # Update step
                if step % self.args.gradient_accumulation_steps == 0:
                    output.loss = output.loss / self.args.gradient_accumulation_steps
                    if self.temp_scheduler is not None:
                        self.tb_writer.add_scalar('Temp', self.temp_scheduler.temp(), self.global_step)
                    self.tb_writer.add_scalar('Loss/train', output.loss, self.global_step)
                    # Backpropogate accumulated loss
                    if self.args.fp16:
                        with amp.scale_loss(output.loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                            self.tb_writer.add_scalar('Scaled_Loss/train', scaled_loss, self.global_step)
                    else:
                        output.loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # Get learning rate
                    self.tb_writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)

                    if output.belief_state_summary:
                        for slot, stats_slot in output.belief_state_summary.items():
                            for key, item in stats_slot.items():
                                self.tb_writer.add_scalar(f'{key}_{slot}/Train', item, self.global_step)

                    # Update model parameters
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    if self.temp_scheduler is not None:
                        self.temp_scheduler.step()

                    tr_loss += output.loss.float().item()
                    epoch_iterator.set_postfix(loss=output.loss.float().item())
                    self.global_step += 1

                # Save model checkpoint
                if self.global_step % self.args.save_steps == 0:
                    logging_loss = tr_loss - logging_loss

                    # Evaluate model
                    if self.args.do_eval:
                        self.eval_mode()
                        metrics = self.evaluate(is_train=True)
                        metrics.training_loss = logging_loss / self.args.save_steps
                        # Log model eval information
                        self.log_info(metrics)
                        self.training_mode()
                    else:
                        metrics = Metrics(training_loss=logging_loss / self.args.save_steps,
                                          joint_goal_accuracy=0.0)
                        self.log_info(metrics)

                    logging_loss = tr_loss

                    try:
                        # Compute the score of the best model
                        self.best_model.compute_score(request=self.model.config.user_request_loss_weight,
                                                      active_domain=self.model.config.active_domain_loss_weight,
                                                      general_act=self.model.config.user_general_act_loss_weight)

                        # Compute the score of the current model
                        metrics.compute_score(request=self.model.config.user_request_loss_weight,
                                              active_domain=self.model.config.active_domain_loss_weight,
                                              general_act=self.model.config.user_general_act_loss_weight)
                    except AttributeError:
                        self.best_model.compute_score()
                        metrics.compute_score()

                    metrics.training_loss = tr_loss / self.global_step

                    if metrics > self.best_model:
                        steps_since_last_update = 0
                        self.logger.info('Model saved.')
                        self.best_model = deepcopy(metrics)

                        self.save_state()
                    else:
                        steps_since_last_update += 1
                        self.logger.info('Model not saved.')

                # Stop training after max training steps or if the model has not updated for too long
                if self.args.max_training_steps > 0 and self.global_step > self.args.max_training_steps:
                    epoch_iterator.close()
                    break
                if self.args.patience > 0 and steps_since_last_update >= self.args.patience:
                    epoch_iterator.close()
                    break

            self.steps_trained_in_current_epoch = 0
            self.logger.info(f'Epoch {e + 1} complete, average training loss = {tr_loss / self.global_step}')

            if self.args.max_training_steps > 0 and self.global_step > self.args.max_training_steps:
                train_iterator.close()
                break
            if self.args.patience > 0 and steps_since_last_update >= self.args.patience:
                train_iterator.close()
                self.logger.info(f'Model has not improved for at least {self.args.patience} steps. Training stopped!')
                break

        # Evaluate final model
        if self.args.do_eval:
            self.eval_mode()
            metrics = self.evaluate(is_train=True)
            metrics.training_loss = tr_loss / self.global_step
            self.log_info(metrics, logging_stage='training_complete')
        else:
            self.logger.info('Training complete!')

        # Store final model
        try:
            self.best_model.compute_score(request=self.model.config.user_request_loss_weight,
                                          active_domain=self.model.config.active_domain_loss_weight,
                                          general_act=self.model.config.user_general_act_loss_weight)

            metrics.compute_score(request=self.model.config.user_request_loss_weight,
                                  active_domain=self.model.config.active_domain_loss_weight,
                                  general_act=self.model.config.user_general_act_loss_weight)
        except AttributeError:
            self.best_model.compute_score()
            metrics.compute_score()

        metrics.training_loss = tr_loss / self.global_step

        if metrics > self.best_model:
            self.logger.info('Final model saved.')
            self.save_state()
        else:
            self.logger.info('Final model not saved, as it is not the best performing model.')

    def evaluate(self, save_eval_path=None, is_train=False, save_pred_dist_path=None, draw_calibration_diagram=False):
        """
        Evaluates the model on the validation set.

        Args:
            save_eval_path (str): Path to save the evaluation results.
            is_train (bool): Whether the evaluation is performed during training.
            save_pred_dist_path (str): Path to save the predicted distribution.
            draw_calibration_diagram (bool): Whether to draw the calibration diagram.
        Returns:
            Metrics: The evaluation metrics.
        """
        save_eval_path = None if is_train else save_eval_path
        save_pred_dist_path = None if is_train else save_pred_dist_path
        draw_calibration_diagram = False if is_train else draw_calibration_diagram
        if not is_train:
            self.logger.info("***** Running evaluation *****")
            self.logger.info("  Num Batches = %d", len(self.validation_dataloader))

        eval_loss = 0.0
        belief_state_summary = dict()
        self.joint_goal_accuracy._init_session()
        self.belief_state_uncertainty_metrics._init_session()
        self.eval_mode(load_slots=True)

        if not is_train:
            epoch_iterator = tqdm(self.validation_dataloader, desc="Iteration")
        else:
            epoch_iterator = self.validation_dataloader
        for batch in epoch_iterator:
            with torch.no_grad():
                input_dict = self.get_input_dict(batch)
                if not is_train and 'distillation' in self.model.config.loss_function:
                    input_dict = {key: input_dict[key] for key in ['input_ids', 'attention_mask', 'token_type_ids']}
                if self.args.ensemble and save_pred_dist_path is not None:
                    input_dict['reduction'] = 'none'
                output = self.model(**input_dict)
                output.loss = output.loss if output.loss is not None else 0.0

            eval_loss += output.loss

            if self.args.ensemble and save_pred_dist_path is not None:
                self.ensemble_aggregator.add_batch(input_dict, output, batch['dialogue_ids'])
                output.belief_state = {slot: probs.mean(-2) for slot, probs in output.belief_state.items()}
                if self.args.predict_actions:
                    output.request_probabilities = {slot: probs.mean(-1)
                                                    for slot, probs in output.request_probabilities.items()}
                    output.active_domain_probabilities = {domain: probs.mean(-1)
                                                        for domain, probs in output.active_domain_probabilities.items()}
                    output.general_act_probabilities = output.general_act_probabilities.mean(-2)

            # Accumulate belief state summary across batches
            if output.belief_state_summary is not None:
                for slot, slot_summary in output.belief_state_summary.items():
                    if slot not in belief_state_summary:
                        belief_state_summary[slot] = dict()
                    for key, item in slot_summary.items():
                        if key not in belief_state_summary[slot]:
                            belief_state_summary[slot][key] = item
                        else:
                            if 'min' in key:
                                belief_state_summary[slot][key] = min(belief_state_summary[slot][key], item)
                            elif 'max' in key:
                                belief_state_summary[slot][key] = max(belief_state_summary[slot][key], item)
                            elif 'mean' in key:
                                belief_state_summary[slot][key] = (belief_state_summary[slot][key] + item) / 2

            slot_0 = [slot for slot in input_dict['state_labels'].keys()] if 'state_labels' in input_dict else list()
            slot_0 = slot_0[0] if slot_0 else None
            if slot_0 is not None:
                pad_dials, pad_turns = torch.where(input_dict['input_ids'][:, :, 0] == -1)
                if len(input_dict['state_labels'][slot_0].size()) == 4:
                    for slot in input_dict['state_labels']:
                        input_dict['state_labels'][slot] = input_dict['state_labels'][slot].mean(-2).argmax(-1)
                        input_dict['state_labels'][slot][pad_dials, pad_turns] = -1
                    if self.args.predict_actions:
                        for slot in input_dict['request_labels']:
                            input_dict['request_labels'][slot] = input_dict['request_labels'][slot].mean(-1).round().int()
                            input_dict['request_labels'][slot][pad_dials, pad_turns] = -1
                        for domain in input_dict['active_domain_labels']:
                            input_dict['active_domain_labels'][domain] = input_dict['active_domain_labels'][domain].mean(-1).round().int()
                            input_dict['active_domain_labels'][domain][pad_dials, pad_turns] = -1
                        input_dict['general_act_labels'] = input_dict['general_act_labels'].mean(-2).argmax(-1)
                        input_dict['general_act_labels'][pad_dials, pad_turns] = -1
            else:
                input_dict = self.get_input_dict(batch)

            # Add batch to metrics
            self.belief_state_uncertainty_metrics.add_dialogues(output.belief_state, input_dict['state_labels'])

            predicted_states = self.tokenizer.decode_state_batch(output.belief_state,
                                                                 output.request_probabilities,
                                                                 output.active_domain_probabilities,
                                                                 output.general_act_probabilities,
                                                                 batch['dialogue_ids'])

            self.joint_goal_accuracy.add_dialogues(predicted_states)

            if self.args.predict_actions:
                self.request_accuracy.add_dialogues(output.request_probabilities, input_dict['request_labels'])
                self.active_domain_accuracy.add_dialogues(output.active_domain_probabilities,
                                                          input_dict['active_domain_labels'])
                self.general_act_accuracy.add_dialogues({'gen': output.general_act_probabilities},
                                                        {'gen': input_dict['general_act_labels']})

        # Compute metrics
        metrics = self.joint_goal_accuracy.evaluate()
        metrics += self.belief_state_uncertainty_metrics.evaluate()
        if self.args.predict_actions:
            metrics += self.request_accuracy.evaluate()
            metrics += self.active_domain_accuracy.evaluate()
            metrics += self.general_act_accuracy.evaluate()
        metrics.validation_loss = eval_loss
        if belief_state_summary:
            metrics.belief_state_summary = belief_state_summary

        # Save model predictions
        if save_eval_path is not None:
            self.joint_goal_accuracy.save_dialogues(save_eval_path)
        if save_pred_dist_path is not None:
            self.ensemble_aggregator.save(save_pred_dist_path)
        if draw_calibration_diagram:
            self.belief_state_uncertainty_metrics.draw_calibration_diagram(
                save_path=self.args.output_dir,
                validation_split=self.joint_goal_accuracy.validation_split
            )

        return metrics
