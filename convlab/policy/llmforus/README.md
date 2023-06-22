# LLMforUS
Training large language model for user simulators.

## Train
We use `accelerate` and `peft` for training large language models.

You should use `accelerate config` to setup training configs, then `accelerate launch accel_train_model.py --model-args` to train the model.

More information is given in `train.sh`