# Dataset Card for CommenGen

- **Repository:** https://github.com/INK-USC/CommonGen
- **Paper:** https://aclanthology.org/2020.findings-emnlp.165.pdf
- **Leaderboard:** https://inklab.usc.edu/CommonGen/leaderboard.html
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

Building machines with commonsense to compose realistically plausible sentences is challenging. CommonGen is a constrained text generation task, associated with a benchmark dataset, to explicitly test machines for the ability of generative commonsense reasoning. Given a set of common concepts; the task is to generate a coherent sentence describing an everyday scenario using these concepts.

CommonGen is challenging because it inherently requires 1) relational reasoning using background commonsense knowledge, and 2) compositional generalization ability to work on unseen concept combinations. Our dataset, constructed through a combination of crowd-sourcing from AMT and existing caption corpora, consists of 30k concept-sets and 50k sentences in total.

- **How to get the transformed data from original data:**
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Set `speaker` to `system`.
  - Retain common concepts annotation in the `concepts` field of each turn.
  - If there are multiple scene sentences in a original sample, split them into multiple samples.
- **Annotations:**
  - concept words

### Supported Tasks and Leaderboards

NLG

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   | non-cat slot span(dialogue act)   |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       67389 |        67389 |         1 |        10.54 |             0 | -                       | -                      | -                              | -                                 |
| validation |        4018 |         4018 |         1 |        11.57 |             0 | -                       | -                      | -                              | -                                 |
| test       |        1497 |         1497 |         1 |         1    |             0 | -                       | -                      | -                              | -                                 |
| all        |       72904 |        72904 |         1 |        10.41 |             0 | -                       | -                      | -                              | -                                 |

0 domains: []
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.


### Citation

```
@inproceedings{lin-etal-2020-commongen,
    title = "{C}ommon{G}en: A Constrained Text Generation Challenge for Generative Commonsense Reasoning",
    author = "Bill Yuchen Lin and Wangchunshu Zhou and Ming Shen and Pei Zhou and Chandra Bhagavatula and Yejin Choi and Xiang Ren",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.165",
}
```

### Licensing Information

MIT License