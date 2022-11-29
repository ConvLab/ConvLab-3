# Dataset Card for CrossWOZ

- **Repository:** https://github.com/thu-coai/CrossWOZ
- **Paper:** https://aclanthology.org/2020.tacl-1.19/
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

CrossWOZ is the first large-scale Chinese Cross-Domain Wizard-of-Oz task-oriented dataset. It contains 6K dialogue sessions and 102K utterances for 5 domains, including hotel, restaurant, attraction, metro, and taxi. Moreover, the corpus contains rich annotation of dialogue states and dialogue acts at both user and system sides. We also provide a user simulator and several benchmark models for pipelined taskoriented dialogue systems, which will facilitate researchers to compare and evaluate their models on this corpus.

- **How to get the transformed data from original data:** 
  - Run `python preprocess.py` in the current directory. Need `../../crosswoz/` as the original data.
- **Main changes of the transformation:**
  - Add simple description for domains, slots, and intents.
  - Switch intent&domain of `General` dialog acts => domain == 'General' and intent in ['thank','bye','greet','welcome']
  - Binary dialog acts include: 1) domain == 'General'; 2) intent in ['NoOffer', 'Request', 'Select']; 3) slot in ['酒店设施']
  - Categorical dialog acts include: slot in ['酒店类型', '车型', '车牌']
  - Non-categorical dialogue acts: others. assert intent in ['Inform', 'Recommend'] and slot != 'none' and value != 'none'
  - Transform original user goal to list of `{domain: {'inform': {slot: [value, mentioned/not mentioned]}, 'request': {slot: [value, mentioned/not mentioned]}}}`, stored as `user_state` of user turns.
  - Transform `sys_state_init` (first API call of system turns) without `selectedResults` as belief state in user turns.
  - Transform `sys_state` (last API call of system turns) to `db_query` with domain states that contain non-empty `selectedResults`. The `selectedResults` are saved as `db_results` (only contain entity name). Both stored in system turns.
- **Annotations:**
  - user goal, user state, dialogue acts, state, db query, db results.
  - Multiple values in state are separated by spaces, meaning all constraints should be satisfied.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG, E2E, User simulator

### Languages

Chinese

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains |   cat slot match(state) | cat slot match(goal)   |   cat slot match(dialogue act) |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |        5012 |        84674 |     16.89 |        20.55 |          3.02 |                   99.67 | -                      |                            100 |                             94.39 |
| validation |         500 |         8458 |     16.92 |        20.53 |          3.04 |                   99.62 | -                      |                            100 |                             94.36 |
| test       |         500 |         8476 |     16.95 |        20.51 |          3.08 |                   99.61 | -                      |                            100 |                             94.85 |
| all        |        6012 |       101608 |     16.9  |        20.54 |          3.03 |                   99.66 | -                      |                            100 |                             94.43 |

6 domains: ['景点', '餐馆', '酒店', '地铁', '出租', 'General']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@article{zhu2020crosswoz,
  author = {Qi Zhu and Kaili Huang and Zheng Zhang and Xiaoyan Zhu and Minlie Huang},
  title = {Cross{WOZ}: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},
  journal = {Transactions of the Association for Computational Linguistics},
  year = {2020}
}
```

### Licensing Information

Apache License, Version 2.0