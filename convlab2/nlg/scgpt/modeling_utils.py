import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING
import torch.cuda.amp as amp
import transformers
from transformers import GPT2LMHeadModel


# reference: https://pytorch.org/docs/master/notes/amp_examples.html
class AmpGPT2LMHeadModel(GPT2LMHeadModel):
    if TYPE_CHECKING:
        # For IDE's code hinting
        forward = GPT2LMHeadModel.forward
    else:
        def forward(self, *args, **kwargs):
            with amp.autocast():
                return super().forward(*args, **kwargs)


def try_enable_gradient_checkpointing(model: "transformers.modeling_utils.PreTrainedModel"):
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        warnings.warn(f"{type(model)} doesn't support gradient_checkpointing")


class AmpHelper:
    """
    References:
        https://pytorch.org/docs/master/notes/amp_examples.html
    """
    def __init__(self, use_amp=True):
        self.use_amp = use_amp
        self.might_enable_autocast = amp.autocast() if use_amp else nullcontext()
        self.scaler = amp.GradScaler()

    def backward(self, loss):
        if self.use_amp:
            return self.scaler.scale(loss).backward()
        else:
            return loss.backward()

    def step(self, optimizer):
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def might_unscale_(self, optimizer):
        if self.use_amp:
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(optimizer)