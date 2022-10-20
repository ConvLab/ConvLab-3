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

# from typing import Any, Dict, List, Optional, Tuple, Union
# from dataclasses import dataclass, field
# import torch
# from torch import nn

# from transformers.deepspeed import is_deepspeed_zero3_enabled
# from transformers.utils import logging, cached_property, torch_required
from transformers.training_args import (
    os, 
    torch,
    logging, 
    dataclass, 
    field, 
    Optional, 
    cached_property, 
    torch_required, 
    get_int_from_env,
    is_torch_tpu_available,
    is_sagemaker_mp_enabled,
    is_sagemaker_dp_enabled,
)

from transformers.trainer import (
    nn,
    Any, Dict, List, Tuple, Union,
    is_deepspeed_zero3_enabled
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datetime import timedelta
import torch.distributed as dist

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

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                if self.xpu_backend == "ccl" and int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1:
                    raise ValueError(
                        "CPU distributed training backend is ccl. but CCL_WORKER_COUNT is not correctly set. "
                        "Please use like 'export CCL_WORKER_COUNT = 1' to set."
                    )

                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                torch.distributed.init_process_group(backend=self.xpu_backend, rank=rank, world_size=size, timeout=timedelta(days=365))
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            dist.init_process_group(backend="smddp", timeout=timedelta(days=365))
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from transformers.deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", timeout=timedelta(days=365))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


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
