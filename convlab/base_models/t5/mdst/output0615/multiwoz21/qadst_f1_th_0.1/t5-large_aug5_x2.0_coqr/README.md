---
tags:
- generated_from_trainer
model-index:
- name: t5-large_aug5_x2.0_coqr
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5-large_aug5_x2.0_coqr

This model is a fine-tuned version of [output0615/multiwoz21/qadst_f1_th>0.1/t5-large_aug5_x2.0_coqr](https://huggingface.co/output0615/multiwoz21/qadst_f1_th>0.1/t5-large_aug5_x2.0_coqr) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 512
- total_eval_batch_size: 256
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.20.1
- Pytorch 1.11.0+cu113
- Datasets 2.6.1
- Tokenizers 0.12.1
