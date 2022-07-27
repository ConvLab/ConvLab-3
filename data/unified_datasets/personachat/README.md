# Dataset Card for Persona-Chat

- **Repository:** https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat
- **Paper:** https://arxiv.org/pdf/1801.07243.pdf
- **Leaderboard:** http://convai.io/2018/
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

We present the PERSONA-CHAT dataset, a new dialogue dataset consisting of 162,064 utterances between crowdworkers who were randomly paired and each asked to act the part of a given provided persona (randomly assigned, and created by another set of crowdworkers). The paired workers were asked to chat naturally and to get to know each other during the conversation. This produces interesting and engaging conversations that our agents can try to learn to mimic.

- **How to get the transformed data from original data:** 
  - download `[train|valid|test]_both_original.txt` from parlai and save them under `original_data` directory.
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Normalize whitespace.
- **Annotations:**
  - Persona, candidate responses of system side

### Supported Tasks and Leaderboards

Response selection, Response generation, Profile prediction

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   | non-cat slot span(dialogue act)   |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |        8939 |       131438 |     14.7  |        10.15 |             0 | -                       | -                      | -                              | -                                 |
| validation |        1000 |        15602 |     15.6  |        10.33 |             0 | -                       | -                      | -                              | -                                 |
| test       |         968 |        15024 |     15.52 |        10.22 |             0 | -                       | -                      | -                              | -                                 |
| all        |       10907 |       162064 |     14.86 |        10.17 |             0 | -                       | -                      | -                              | -                                 |

0 domains: []
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{zhang-etal-2018-personalizing,
    title = "Personalizing Dialogue Agents: {I} have a dog, do you have pets too?",
    author = "Zhang, Saizheng and Dinan, Emily and Urbanek, Jack and Szlam, Arthur and Kiela, Douwe and Weston, Jason",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2018",
    publisher = "Association for Computational Linguistics",
}
```

### Licensing Information

TODO