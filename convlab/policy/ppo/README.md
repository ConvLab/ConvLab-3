# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (Schulmann et. al. 2017) is an on-policy reinforcement learning algorithm. The architecture used is a simple MLP and thus not transferable to new ontologies.

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
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}

@inproceedings{zhu-etal-2020-convlab,
    title = "{C}onv{L}ab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems",
    author = "Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-demos.19",
    doi = "10.18653/v1/2020.acl-demos.19",
    pages = "142--149"
}
```