# Policy Gradient (PG)

PG is an on-policy reinforcement learning algorithm that uses the policy gradient theorem to perform policy updates, using directly the return as value estimation
. 
## Supervised pre-training

If you want to obtain a supervised model for pre-training, please have a look in the MLE policy folder.

## RL training

Starting a RL training is as easy as executing

```sh
$ python train.py --path=your_environment_config --seed=SEED
```

One example for the environment-config is **semantic_level_config.json**, where parameters for the training are specified, for instance

- load_path: provide a path to initialise the model with a pre-trained model, skip the ending .pol.mdl
- process_num: the number of processes to use during evaluation to speed it up
- num_eval_dialogues: how many evaluation dialogues should be used
- epoch: how many training epochs to run. One epoch consists of collecting dialogues + performing an update
- eval_frequency: after how many epochs perform an evaluation
- batchsz: the number of training dialogues collected before doing an update

Moreover, you can specify the full dialogue pipeline here, such as the user policy, NLU for system and user, etc.

Parameters that are tied to the RL algorithm and the model architecture can be changed in config.json.


## Evaluation

For creating evaluation plots and running evaluation dialogues, please have a look in the README of the policy folder.

## References

```
@inproceedings{NIPS1999_464d828b,
 author = {Sutton, Richard S and McAllester, David and Singh, Satinder and Mansour, Yishay},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Solla and T. Leen and K. M\"{u}ller},
 pages = {},
 publisher = {MIT Press},
 title = {Policy Gradient Methods for Reinforcement Learning with Function Approximation},
 url = {https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf},
 volume = {12},
 year = {1999}
}
```