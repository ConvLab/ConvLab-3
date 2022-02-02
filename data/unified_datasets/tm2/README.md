# Dataset Card for Taskmaster-1

- **Repository:** https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020
- **Paper:** https://arxiv.org/pdf/1909.05358.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

The Taskmaster-2 dataset consists of 17,289 dialogs in the seven domains. Unlike Taskmaster-1, which includes both written "self-dialogs" and spoken two-person dialogs, Taskmaster-2 consists entirely of spoken two-person dialogs. In addition, while Taskmaster-1 is almost exclusively task-based, Taskmaster-2 contains a good number of search- and recommendation-oriented dialogs, as seen for example in the restaurants, flights, hotels, and movies verticals. The music browsing and sports conversations are almost exclusively search- and recommendation-based. All dialogs in this release were created using a Wizard of Oz (WOz) methodology in which crowdsourced workers played the role of a 'user' and trained call center operators played the role of the 'assistant'. In this way, users were led to believe they were interacting with an automated system that “spoke” using text-to-speech (TTS) even though it was in fact a human behind the scenes. As a result, users could express themselves however they chose in the context of an automated interface.

- **How to get the transformed data from original data:** 
  - Download [master.zip](https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip).
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Remove dialogs that are empty or only contain one speaker.
  - Split each domain dialogs into train/validation/test randomly (8:1:1).
  - Merge continuous turns by the same speaker (ignore repeated turns).
  - Annotate `dialogue acts` according to the original segment annotations. Add `intent` annotation (`==inform`). The type of `dialogue act` is set to `non-categorical` if the `slot` is not in `anno2slot` in `preprocess.py`). Otherwise, the type is set to `binary` (and the `value` is empty). If there are multiple spans overlapping, we only keep the shortest one, since we found that this simple strategy can reduce the noise in annotation.
  - Add `domain`, `intent`, and `slot` descriptions.
  - Add `state` by accumulate `non-categorical dialogue acts` in the order that they appear.
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
| train      |       13838 |       234321 |     16.93 |         9.1  |             1 | -                       | -                      | -                              |                               100 |
| validation |        1731 |        29349 |     16.95 |         9.15 |             1 | -                       | -                      | -                              |                               100 |
| test       |        1734 |        29447 |     16.98 |         9.07 |             1 | -                       | -                      | -                              |                               100 |
| all        |       17303 |       293117 |     16.94 |         9.1  |             1 | -                       | -                      | -                              |                               100 |

7 domains: ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']
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