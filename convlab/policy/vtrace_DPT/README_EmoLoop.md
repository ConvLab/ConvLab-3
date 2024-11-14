# Dynamic Dialogue Policy Transformer (DDPT) for EmoLoop - EmoDDPT

The dynamic dialogue policy transformer (Geishauser et. al. 2022) is a model built for continual reinforcement learning. It uses a pre-trained RoBERTa language model to construct embeddings for each state information and domain, slot and value in the action set. As a consequence, it can be used for different ontologies and is able to deal with new state information as well as actions. The backbone architecture is a transformer encoder-decoder.

It uses the CLEAR algorithm (Rolnick et. al. 2019) for continual reinforcement learning that builds on top of VTRACE (Espheholt et. al. 2018). The current folder supports only training in a stationary environment and no continual learning, which uses VTRACE as algorithm.

This README contains instructions for running EmoDDPT, a variant of DDPT that is adapted for EmoLoop (Feng et al., 2024) to additionally take user emotion as input and system conduct as output.

## Supervised pre-training

If you want to pre-train the model on a dataset, use the command

```sh
$ python supervised_emo_conduct/train_supervised.py --dataset_name=emowoz --seed=SEED --model_path="" --use_emotion
```

The first time you run that command, it will take longer as the dataset needs to be pre-processed.

This will create a corresponding experiments folder under supervised/experiments, where the model is saved in /save.

You can specify hyperparamters such as epoch, supervised_lr and data_percentage (how much of the data you want to use) in the config.json file.

## RL training

Starting a RL training is as easy as executing

```sh
$ python train.py --config_name=configs/emoloop_pipeline_config.json --hyperparameter=configs/emoloop_hyperparameters.json --seed=SEED
```

The environment-config for EmoLoop is **configs/emoloop_pipeline_config**, where modules and parameters for the training are specified, for instance

- load_path: provide a path to initialise the model with a pre-trained model, skip the ending .pol.mdl
- process_num: the number of processes to use during evaluation to speed it up
- num_eval_dialogues: how many evaluation dialogues should be used
- eval_frequency: after how many training dialogues an evaluation should be performed
- total_dialogues: how many training dialogues should be done in total
- new_dialogues: how many new dialogues should be collected before a policy update

Moreover, you can specify the full dialogue pipeline here, such as the user policy, NLU for system and user, etc. Please specify paths to model checkpoints in the configuration file for each module.

Parameters that are tied to the RL algorithm and the model architecture can be changed in **configs/emoloop_hyperparameters.json**.

NOTE: you can specify which underlying dataset should be used for creating the action and state space through changing in your **environment-config**


## Evaluation

For creating evaluation plots and running evaluation dialogues, please have a look in the README of the policy folder.

## Interface

To use trained models in a dialog system, import them through:

```python
from convlab.policy.vector.vector_nodes import VectorNodes
from convlab.policy.vtrace_DPT import VTRACE

vectorizer = VectorNodes(dataset_name='multiwoz21',
                         use_masking=False,
                         manually_add_entity_names=True,
                         seed=0,
                         filter_state=True)
ddpt = VTRACE(is_train=True,
              seed=0,
              vectorizer=vectorizer,
              load_path="ddpt")
```
Specify the appropriate load_path in VTRACE.

## References

```
@inproceedings{geishauser-etal-2022-dynamic,
    title = "Dynamic Dialogue Policy for Continual Reinforcement Learning",
    author = "Geishauser, Christian  and
      van Niekerk, Carel  and
      Lin, Hsien-chin  and
      Lubis, Nurul  and
      Heck, Michael  and
      Feng, Shutong  and
      Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.21",
    pages = "266--284",
    abstract = "Continual learning is one of the key components of human learning and a necessary requirement of artificial intelligence. As dialogue can potentially span infinitely many topics and tasks, a task-oriented dialogue system must have the capability to continually learn, dynamically adapting to new challenges while preserving the knowledge it already acquired. Despite the importance, continual reinforcement learning of the dialogue policy has remained largely unaddressed. The lack of a framework with training protocols, baseline models and suitable metrics, has so far hindered research in this direction. In this work we fill precisely this gap, enabling research in dialogue policy optimisation to go from static to dynamic learning. We provide a continual learning algorithm, baseline architectures and metrics for assessing continual learning models. Moreover, we propose the dynamic dialogue policy transformer (DDPT), a novel dynamic architecture that can integrate new knowledge seamlessly, is capable of handling large state spaces and obtains significant zero-shot performance when being exposed to unseen domains, without any growth in network parameter size. We validate the strengths of DDPT in simulation with two user simulators as well as with humans.",
}

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

@InProceedings{pmlr-v80-espeholt18a,
  title = 	 {{IMPALA}: Scalable Distributed Deep-{RL} with Importance Weighted Actor-Learner Architectures},
  author =       {Espeholt, Lasse and Soyer, Hubert and Munos, Remi and Simonyan, Karen and Mnih, Vlad and Ward, Tom and Doron, Yotam and Firoiu, Vlad and Harley, Tim and Dunning, Iain and Legg, Shane and Kavukcuoglu, Koray},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1407--1416},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf},
  url = 	 {https://proceedings.mlr.press/v80/espeholt18a.html},
  abstract = 	 {In this work we aim to solve a large collection of tasks using a single reinforcement learning agent with a single set of parameters. A key challenge is to handle the increased amount of data and extended training time. We have developed a new distributed agent IMPALA (Importance Weighted Actor-Learner Architecture) that not only uses resources more efficiently in single-machine training but also scales to thousands of machines without sacrificing data efficiency or resource utilisation. We achieve stable learning at high throughput by combining decoupled acting and learning with a novel off-policy correction method called V-trace. We demonstrate the effectiveness of IMPALA for multi-task reinforcement learning on DMLab-30 (a set of 30 tasks from the DeepMind Lab environment (Beattie et al., 2016)) and Atari57 (all available Atari games in Arcade Learning Environment (Bellemare et al., 2013a)). Our results show that IMPALA is able to achieve better performance than previous agents with less data, and crucially exhibits positive transfer between tasks as a result of its multi-task approach.}
}

@inproceedings{feng-etal-2024-infusing,
    title = "Infusing Emotions into Task-oriented Dialogue Systems: Understanding, Management, and Generation",
    author = "Feng, Shutong  and
      Lin, Hsien-chin  and
      Geishauser, Christian  and
      Lubis, Nurul  and
      van Niekerk, Carel  and
      Heck, Michael  and
      Ruppik, Benjamin Matthias  and
      Vukovic, Renato  and
      Gasic, Milica",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.60",
    doi = "10.18653/v1/2024.sigdial-1.60",
    pages = "699--717",
    abstract = "Emotions are indispensable in human communication, but are often overlooked in task-oriented dialogue (ToD) modelling, where the task success is the primary focus. While existing works have explored user emotions or similar concepts in some ToD tasks, none has so far included emotion modelling into a fully-fledged ToD system nor conducted interaction with human or simulated users. In this work, we incorporate emotion into the complete ToD processing loop, involving understanding, management, and generation. To this end, we extend the EmoWOZ dataset (Feng et al., 2022) with system affective behaviour labels. Through interactive experimentation involving both simulated and human users, we demonstrate that our proposed framework significantly enhances the user{'}s emotional experience as well as the task success.",
}

```