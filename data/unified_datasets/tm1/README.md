# Dataset Card for Taskmaster-1

- **Repository:** https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019
- **Paper:** https://arxiv.org/pdf/1909.05358.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

The original dataset consists of 13,215 task-based dialogs, including 5,507 spoken and 7,708 written dialogs created with two distinct procedures. Each conversation falls into one of six domains: ordering pizza, creating auto repair appointments, setting up ride service, ordering movie tickets, ordering coffee drinks and making restaurant reservations.

- **How to get the transformed data from original data:** 
  - Download [master.zip](https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip).
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Remove dialogs that are empty or only contain one speaker.
  - Split woz-dialogs into train/validation/test randomly (8:1:1). The split of self-dialogs is followed the original dataset.
  - Merge continuous turns by the same speaker (ignore repeated turns).
  - Annotate `dialogue acts` according to the original segment annotations. Add `intent` annotation (inform/accept/reject). The type of `dialogue act` is set to `non-categorical` if the original segment annotation includes a specified `slot`. Otherwise, the type is set to `binary` (and the `slot` and `value` are empty) since it means general reference to a transaction, e.g. "OK your pizza has been ordered".
  - Add `intent` and `slot` descriptions.
  - Add `state` by accumulate dialog acts except those whose intents are **reject**.
  - Keep the first annotation since each conversation was annotated by two workers.
- **Annotations:**
  - dialogue acts, state.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       10535 |       223322 |     21.2  |         8.75 |             1 | -                       | -                      | -                              |                               100 |
| validation |        1318 |        27903 |     21.17 |         8.75 |             1 | -                       | -                      | -                              |                               100 |
| test       |        1322 |        27660 |     20.92 |         8.87 |             1 | -                       | -                      | -                              |                               100 |
| all        |       13175 |       278885 |     21.17 |         8.76 |             1 | -                       | -                      | -                              |                               100 |

6 domains: ['uber_lyft', 'movie_ticket', 'restaurant_reservation', 'coffee_ordering', 'pizza_ordering', 'auto_repair']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{byrne-etal-2019-taskmaster,
  title = {Taskmaster-1:Toward a Realistic and Diverse Dialog Dataset},
  author = {Bill Byrne and Karthik Krishnamoorthi and Chinnadhurai Sankar and Arvind Neelakantan and Daniel Duckworth and Semih Yavuz and Ben Goodrich and Amit Dubey and Kyu-Young Kim and Andy Cedilnik},
  booktitle = {2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing},
  address = {Hong Kong}, 
  year = {2019} 
}
```

### Licensing Information

[**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/)