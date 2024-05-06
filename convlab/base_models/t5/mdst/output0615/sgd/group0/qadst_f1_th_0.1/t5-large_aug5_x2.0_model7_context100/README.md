---
tags:
- generated_from_trainer
model-index:
- name: t5-large_aug5_x2.0_model7_context100
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5-large_aug5_x2.0_model7_context100

This model is a fine-tuned version of [/zhangpai23/zhuqi/pre-trained-models/t5-large](https://huggingface.co//zhangpai23/zhuqi/pre-trained-models/t5-large) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1199

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
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0356        | 1.0   | 1822 | 0.1016          |
| 0.0115        | 2.0   | 3644 | 0.0916          |
| 0.0049        | 3.0   | 5466 | 0.0792          |
| 0.0014        | 4.0   | 7288 | 0.0953          |
| 0.0004        | 5.0   | 9110 | 0.1199          |


### Framework versions

- Transformers 4.20.1
- Pytorch 1.11.0+cu113
- Datasets 2.6.1
- Tokenizers 0.12.1
