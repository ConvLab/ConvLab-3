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
- Loss: 0.1918

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
| 0.0613        | 1.0   | 1528 | 0.1145          |
| 0.0284        | 2.0   | 3056 | 0.1073          |
| 0.0133        | 3.0   | 4584 | 0.1304          |
| 0.0053        | 4.0   | 6112 | 0.1529          |
| 0.0016        | 5.0   | 7640 | 0.1918          |


### Framework versions

- Transformers 4.20.1
- Pytorch 1.11.0+cu113
- Datasets 2.6.1
- Tokenizers 0.12.1
