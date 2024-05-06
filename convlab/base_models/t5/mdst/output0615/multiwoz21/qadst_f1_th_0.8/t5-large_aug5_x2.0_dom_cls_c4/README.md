---
tags:
- generated_from_trainer
model-index:
- name: t5-large_aug5_x2.0_dom_cls_c4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5-large_aug5_x2.0_dom_cls_c4

This model is a fine-tuned version of [output0615/multiwoz21/qadst_f1_th>0.8/t5-large_aug5_x2.0_dom_cls_c4](https://huggingface.co/output0615/multiwoz21/qadst_f1_th>0.8/t5-large_aug5_x2.0_dom_cls_c4) on an unknown dataset.

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
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 128
- total_eval_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.20.1
- Pytorch 1.11.0+cu113
- Datasets 2.6.1
- Tokenizers 0.12.1
