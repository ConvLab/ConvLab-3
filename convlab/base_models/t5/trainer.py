# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from email.policy import default
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging, add_start_docstrings
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


logger = logging.get_logger(__name__)

@dataclass
class ConvLabSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    `ConvLabSeq2SeqTrainingArguments` is a subclass of `Seq2SeqTrainingArguments` that adds the
    following arguments: `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, and
    `num_return_sequences`
    """
    do_sample: bool = field(default=False, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "The value used to module the next token probabilities."})
    top_k: Optional[int] = field(default=0, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."})
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "The number of independently computed returned sequences for each element in the batch."})



class ConvLabSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "do_sample": self.args.do_sample,
            "temperature": self.args.temperature,
            "top_k": self.args.top_k,
            "top_p": self.args.top_p,
            "num_return_sequences": self.args.num_return_sequences
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
