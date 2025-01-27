# Emotion Dialogue State Tracker
This module is implemented as a wrapper consisting of an emotion recogniser and an ordinary dialogue state tracker. Therefore, to initialise the wrapper, both an emotion recogniser and dialogue state tracker are needed.

## Emotion Recogniser
See `modeling/` for training and inferencing.

## Dialogue State Tracker
SetSUMBT (van Niekerk et al., 2021), Trippy (Heck et al., 2020), and BERTNLU+RuleDST (Zhu et al., 2020) have been implemented and tested on the individual module level. Only SetSumbt was fully tested and used in the interactive loop. Respective model checkpoints can be obtained from [ConvLab-3 Huggingface homepage](https://huggingface.co/ConvLab).

## Testing Emotion Dialogue State Tracker
See `test.py` to initialise and inference with the emotion dialogue state tracker.

## References

```
@inproceedings{van-niekerk-etal-2021-uncertainty,
    title = "Uncertainty Measures in Neural Belief Tracking and the Effects on Dialogue Policy Performance",
    author = "van Niekerk, Carel  and
      Malinin, Andrey  and
      Geishauser, Christian  and
      Heck, Michael  and
      Lin, Hsien-chin  and
      Lubis, Nurul  and
      Feng, Shutong  and
      Gasic, Milica",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.623",
    doi = "10.18653/v1/2021.emnlp-main.623",
    pages = "7901--7914",
    abstract = "The ability to identify and resolve uncertainty is crucial for the robustness of a dialogue system. Indeed, this has been confirmed empirically on systems that utilise Bayesian approaches to dialogue belief tracking. However, such systems consider only confidence estimates and have difficulty scaling to more complex settings. Neural dialogue systems, on the other hand, rarely take uncertainties into account. They are therefore overconfident in their decisions and less robust. Moreover, the performance of the tracking task is often evaluated in isolation, without consideration of its effect on the downstream policy optimisation. We propose the use of different uncertainty measures in neural belief tracking. The effects of these measures on the downstream task of policy optimisation are evaluated by adding selected measures of uncertainty to the feature space of the policy and training policies through interaction with a user simulator. Both human and simulated user results show that incorporating these measures leads to improvements both of the performance and of the robustness of the downstream dialogue policy. This highlights the importance of developing neural dialogue belief trackers that take uncertainty into account.",
}
```

```
@inproceedings{heck-etal-2020-trippy,
    title = "{T}rip{P}y: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking",
    author = "Heck, Michael  and
      van Niekerk, Carel  and
      Lubis, Nurul  and
      Geishauser, Christian  and
      Lin, Hsien-Chin  and
      Moresi, Marco  and
      Gasic, Milica",
    editor = "Pietquin, Olivier  and
      Muresan, Smaranda  and
      Chen, Vivian  and
      Kennington, Casey  and
      Vandyke, David  and
      Dethlefs, Nina  and
      Inoue, Koji  and
      Ekstedt, Erik  and
      Ultes, Stefan",
    booktitle = "Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.sigdial-1.4",
    doi = "10.18653/v1/2020.sigdial-1.4",
    pages = "35--44",
    abstract = "Task-oriented dialog systems rely on dialog state tracking (DST) to monitor the user{'}s goal during the course of an interaction. Multi-domain and open-vocabulary settings complicate the task considerably and demand scalable solutions. In this paper we present a new approach to DST which makes use of various copy mechanisms to fill slots with values. Our model has no need to maintain a list of candidate values. Instead, all values are extracted from the dialog context on-the-fly. A slot is filled by one of three copy mechanisms: (1) Span prediction may extract values directly from the user input; (2) a value may be copied from a system inform memory that keeps track of the system{'}s inform operations (3) a value may be copied over from a different slot that is already contained in the dialog state to resolve coreferences within and across domains. Our approach combines the advantages of span-based slot filling methods with memory methods to avoid the use of value picklists altogether. We argue that our strategy simplifies the DST task while at the same time achieving state of the art performance on various popular evaluation sets including Multiwoz 2.1, where we achieve a joint goal accuracy beyond 55{\%}.",
}
```

```
@inproceedings{zhu-etal-2020-convlab,
    title = "{C}onv{L}ab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems",
    author = "Zhu, Qi  and
      Zhang, Zheng  and
      Fang, Yan  and
      Li, Xiang  and
      Takanobu, Ryuichi  and
      Li, Jinchao  and
      Peng, Baolin  and
      Gao, Jianfeng  and
      Zhu, Xiaoyan  and
      Huang, Minlie",
    editor = "Celikyilmaz, Asli  and
      Wen, Tsung-Hsien",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-demos.19",
    doi = "10.18653/v1/2020.acl-demos.19",
    pages = "142--149",
    abstract = "We present ConvLab-2, an open-source toolkit that enables researchers to build task-oriented dialogue systems with state-of-the-art models, perform an end-to-end evaluation, and diagnose the weakness of systems. As the successor of ConvLab, ConvLab-2 inherits ConvLab{'}s framework but integrates more powerful dialogue models and supports more datasets. Besides, we have developed an analysis tool and an interactive tool to assist researchers in diagnosing dialogue systems. The analysis tool presents rich statistics and summarizes common mistakes from simulated dialogues, which facilitates error analysis and system improvement. The interactive tool provides an user interface that allows developers to diagnose an assembled dialogue system by interacting with the system and modifying the output of each system component.",
}
```