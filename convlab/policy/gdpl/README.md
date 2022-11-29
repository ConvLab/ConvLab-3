# Guided Dialogue Policy Learning (GDPL)

GDPL uses the PPO algorithm to optimize the policy. The difference to vanilla PPO is that it is not using the extrinsic reward for optimization but leverages inverse reinforcement learning to train a reward estimator. This reward estimator provides the reward that should be optimized.

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
@inproceedings{takanobu-etal-2019-guided,
    title = "Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog",
    author = "Takanobu, Ryuichi  and
      Zhu, Hanlin  and
      Huang, Minlie",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1010",
    doi = "10.18653/v1/D19-1010",
    pages = "100--110",
    abstract = "Dialog policy decides what and how a task-oriented dialog system will respond, and plays a vital role in delivering effective conversations. Many studies apply Reinforcement Learning to learn a dialog policy with the reward function which requires elaborate design and pre-specified user goals. With the growing needs to handle complex goals across multiple domains, such manually designed reward functions are not affordable to deal with the complexity of real-world tasks. To this end, we propose Guided Dialog Policy Learning, a novel algorithm based on Adversarial Inverse Reinforcement Learning for joint reward estimation and policy optimization in multi-domain task-oriented dialog. The proposed approach estimates the reward signal and infers the user goal in the dialog sessions. The reward estimator evaluates the state-action pairs so that it can guide the dialog policy at each dialog turn. Extensive experiments on a multi-domain dialog dataset show that the dialog policy guided by the learned reward function achieves remarkably higher task success than state-of-the-art baselines.",
}
```