# Dataset Card for WOZ 2.0

- **Repository:** https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz
- **Paper:** https://aclanthology.org/P17-1163.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

Describe the dataset.

- **How to get the transformed data from original data:** 
  - download `woz_[train|validate|test]_en.json` from https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz and save to `woz` dir in the current directory.
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - domain is set to **restaurant**.
  - normalize the value of categorical slots in state and dialogue acts.
  - `belief_states` in WOZ dataset contains `request` intents, which are ignored in processing.
  - use simple string match to find value spans of non-categorical slots.

- **Annotations:**
  - User dialogue acts, state

### Supported Tasks and Leaderboards

NLU, DST, E2E

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains |   cat slot match(state) | cat slot match(goal)   |   cat slot match(dialogue act) |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |         600 |         4472 |      7.45 |        11.37 |             1 |                     100 | -                      |                            100 |                             96.56 |
| validation |         200 |         1460 |      7.3  |        11.28 |             1 |                     100 | -                      |                            100 |                             95.52 |
| test       |         400 |         2892 |      7.23 |        11.49 |             1 |                     100 | -                      |                            100 |                             94.83 |
| all        |        1200 |         8824 |      7.35 |        11.39 |             1 |                     100 | -                      |                            100 |                             95.83 |

1 domains: ['restaurant']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{mrksic-etal-2017-neural,
    title = "Neural Belief Tracker: Data-Driven Dialogue State Tracking",
    author = "Mrk{\v{s}}i{\'c}, Nikola  and
      {\'O} S{\'e}aghdha, Diarmuid  and
      Wen, Tsung-Hsien  and
      Thomson, Blaise  and
      Young, Steve",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1163",
    doi = "10.18653/v1/P17-1163",
    pages = "1777--1788",
}
```

### Licensing Information

Apache License, Version 2.0
