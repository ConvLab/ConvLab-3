# Dataset Card for KVRET

- **Repository:** https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- **Paper:** https://arxiv.org/pdf/1705.05414.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

In an effort to help alleviate this problem, we release a corpus of 3,031 multi-turn dialogues in three distinct domains appropriate for an in-car assistant: calendar scheduling, weather information retrieval, and point-of-interest navigation. Our dialogues are grounded through knowledge bases ensuring that they are versatile in their natural language without being completely free form.

- **How to get the transformed data from original data:**
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Create user `dialogue acts` and `state` according to original annotation.
  - Put dialogue level kb into system side `db_results`.
  - Skip repeated turns and empty dialogue.
- **Annotations:**
  - user dialogue acts, state, db_results.

### Supported Tasks and Leaderboards

NLU, DST, Context-to-response

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |        2424 |        12720 |      5.25 |         8.02 |             1 | -                       | -                      | -                              |                             98.07 |
| validation |         302 |         1566 |      5.19 |         7.93 |             1 | -                       | -                      | -                              |                             97.62 |
| test       |         304 |         1627 |      5.35 |         7.7  |             1 | -                       | -                      | -                              |                             97.72 |
| all        |        3030 |        15913 |      5.25 |         7.98 |             1 | -                       | -                      | -                              |                             97.99 |

3 domains: ['schedule', 'weather', 'navigate']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.


### Citation

```
@inproceedings{eric-etal-2017-key,
    title = "Key-Value Retrieval Networks for Task-Oriented Dialogue",
    author = "Eric, Mihail  and
      Krishnan, Lakshmi  and
      Charette, Francois  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
    year = "2017",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5506",
}
```

### Licensing Information

TODO