# Maximum Likelihood Estimator (MLE)

MLE learns a MLP model in a supervised way using a provided dataset. The trained model can be used as intialisation point for running RL trainings with PPO or GDPL for instance.

## Supervised Training

Starting a training is as easy as executing

```sh
$ python train.py --dataset_name=DATASET_NAME --seed=SEED --eval_freq=FREQ
```

The dataset name can be "multiwoz21" or "sgd" for instance. The first time you run that command, it will take longer as the dataset needs to be pre-processed. The evaluation frequency decides after how many epochs should be evaluated.

Other hyperparameters such as learning rate or number of epochs can be set in the config.json file.

We provide a model trained on multiwoz21 on hugging-face: https://huggingface.co/ConvLab/mle-policy-multiwoz21


## Evaluation

Evaluation on the validation data set takes place during training.