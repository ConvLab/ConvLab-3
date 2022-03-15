# Dataset Card for Camrest

- **Repository:** https://www.repository.cam.ac.uk/handle/1810/260970
- **Paper:** https://aclanthology.org/D16-1233/
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

Cambridge restaurant dialogue domain dataset collected for developing neural network based dialogue systems. The two papers published based on this dataset are: 1. A Network-based End-to-End Trainable Task-oriented Dialogue System 2. Conditional Generation and Snapshot Learning in Neural Dialogue Systems. The dataset was collected based on the Wizard of Oz experiment on Amazon MTurk. Each dialogue contains a goal label and several exchanges between a customer and the system. Each user turn was labelled by a set of slot-value pairs representing a coarse representation of dialogue state (`slu` field). There are in total 676 dialogue, in which most of the dialogues are finished but some of dialogues were not.

- **How to get the transformed data from original data:** 
  - Run `python preprocess.py` in the current directory. Need `../../camrest/` as the original data.
- **Main changes of the transformation:**
  - Add dialogue act annotation according to the state change. This step was done by ConvLab-2 and we use the processed dialog acts here.
  - Rename `pricerange` to `price range`
  - Add character level span annotation for non-categorical slots.
- **Annotations:**
  - user goal, dialogue acts, state.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG, E2E, User simulator

### Languages

English

### Data Splits

| split      | dialogues | utterances | avg_utt | avg_tokens | avg_domains | cat slot match(state) | cat slot match(goal) | cat slot match(dialogue act) | non-cat slot span(dialogue act) |
| ---------- | --------- | ---------- | ------- | ---------- | ----------- | --------------------- | -------------------- | ---------------------------- | ------------------------------- |
| train      | 406       | 3342       | 8.23    | 10.6       | 1           | 100                   | 100                  | 100                          | 99.83                           |
| validation | 135       | 1076       | 7.97    | 11.26      | 1           | 100                   | 100                  | 100                          | 100                             |
| test       | 135       | 1070       | 7.93    | 11.01      | 1           | 100                   | 100                  | 100                          | 100                             |
| all        | 676       | 5488       | 8.12    | 10.81      | 1           | 100                   | 100                  | 100                          | 99.9                            |

1 domains: ['restaurant']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{wen-etal-2016-conditional,
    title = "Conditional Generation and Snapshot Learning in Neural Dialogue Systems",
    author = "Wen, Tsung-Hsien and Ga{\v{s}}i{\'c}, Milica and Mrk{\v{s}}i{\'c}, Nikola  and Rojas-Barahona, Lina M. and Su, Pei-Hao and Ultes, Stefan and Vandyke, David and Young, Steve",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1233",
    doi = "10.18653/v1/D16-1233",
    pages = "2153--2162",
}
```

### Licensing Information

[**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/)