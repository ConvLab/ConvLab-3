# Dynamic Dialogue Policy Transformer (DDPT)

The dynamic dialogue policy transformer (Geishauser et. al. 2022) is a model built for continual reinforcement learning. It uses a pre-trained RoBERTa language model to construct embeddings for each state information and domain, slot and value in the action set. As a consequence, it can be used for different ontologies and is able to deal with new state information as well as actions. The backbone architecture is a transformer encoder-decoder.

It uses the CLEAR algorithm (Rolnick et. al. 2019) for continual reinforcement learning that builds on top of VTRACE (Espheholt et. al. 2018). The current folder supports only training in a stationary environment and no continual learning, which uses VTRACE as algorithm.

## Supervised pre-training

If you want to pre-train the model on a dataset, use the command

```sh
$ python supervised/train_supervised.py --dataset_name=DATASET_NAME --seed=SEED --model_path=""
```

The first time you run that command, it will take longer as the dataset needs to be pre-processed.

This will create a corresponding experiments folder under supervised/experiments, where the model is saved in /save.

You can specify the dataset that you would like to use, e.g. "multiwoz21" or "sgd". You can also specify a model_path if you have already a pre-trained model, for instance when you first train on SGD before you fine-tune on multiwoz21 data.

You can specify hyperparamters such as epoch, supervised_lr and data_percentage (how much of the data you want to use) in the config.json file.



## RL training

Starting a RL training is as easy as executing

```sh
$ python train.py --path=your_environment_config --seed=SEED
```

One example for the environment-config is **semantic_level_config.json**, where parameters for the training are specified, for instance

- load_path: provide a path to initialise the model with a pre-trained model, skip the ending .pol.mdl
- process_num: the number of processes to use during evaluation to speed it up
- num_eval_dialogues: how many evaluation dialogues should be used
- eval_frequency: after how many training dialogues an evaluation should be performed
- total_dialogues: how many training dialogues should be done in total
- new_dialogues: how many new dialogues should be collected before a policy update

Moreover, you can specify the full dialogue pipeline here, such as the user policy, NLU for system and user, etc.

Parameters that are tied to the RL algorithm and the model architecture can be changed in config.json.


## Evaluation

For creating evaluation plots and running evaluation dialogues, please have a look in the README of the policy folder.

## References

```
@inproceedings{NEURIPS2019_fa7cdfad,
 author = {Rolnick, David and Ahuja, Arun and Schwarz, Jonathan and Lillicrap, Timothy and Wayne, Gregory},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Experience Replay for Continual Learning},
 url = {https://proceedings.neurips.cc/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf},
 volume = {32},
 year = {2019}
}
```