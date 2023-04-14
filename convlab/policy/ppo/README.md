# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (Schulmann et. al. 2017) is an on-policy reinforcement learning algorithm. The architecture used is a simple MLP and thus not transferable to new ontologies.

## Supervised pre-training

If you want to obtain a supervised model for pre-training, please have a look in the MLE policy folder.

## RL training

Starting a RL training is as easy as executing

```sh
$ python train.py --config_name=your_config_name --seed=SEED
```

One example for the environment-config is **RuleUser-Semantic-RuleDST**, where parameters for the training are specified, for instance

- load_path: provide a path to initialise the model with a pre-trained model, skip the ending .pol.mdl
- process_num: the number of processes to use during evaluation to speed it up
- num_eval_dialogues: how many evaluation dialogues should be used
- epoch: how many training epochs to run. One epoch consists of collecting dialogues + performing an update
- eval_frequency: after how many epochs perform an evaluation
- num_train_dialogues: the number of training dialogues collected before doing an update

Moreover, you can specify the full dialogue pipeline here, such as the user policy, NLU for system and user, etc.

Parameters that are tied to the RL algorithm and the model architecture can be changed in config.json.


## Evaluation

For creating evaluation plots and running evaluation dialogues, please have a look in the README of the policy folder.

## References

```
@article{DBLP:journals/corr/SchulmanWDRK17,
  author    = {John Schulman and
               Filip Wolski and
               Prafulla Dhariwal and
               Alec Radford and
               Oleg Klimov},
  title     = {Proximal Policy Optimization Algorithms},
  journal   = {CoRR},
  volume    = {abs/1707.06347},
  year      = {2017},
  url       = {http://arxiv.org/abs/1707.06347},
  eprinttype = {arXiv},
  eprint    = {1707.06347},
  timestamp = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```