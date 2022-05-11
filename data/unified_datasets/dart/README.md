# Dataset Card for DailyDialog

- **Repository:** https://github.com/Yale-LILY/dart
- **Paper:** https://arxiv.org/pdf/2007.02871.pdf
- **Leaderboard:** https://github.com/Yale-LILY/dart
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

DART is a large and open-domain structured DAta Record to Text generation corpus with high-quality sentence annotations with each input being a set of entity-relation triples following a tree-structured ontology. It consists of 82191 examples across different domains with each input being a semantic triple set derived from data records in tables and the tree ontology of table schema, annotated with sentence description that covers all facts in the triple set.

- **How to get the transformed data from original data:** 
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Use `source` annotation as `domain`.
  - Retain entity-relation triples in the `tripleset` field of each turn.
  - If there are multiple source&text annotation in a original sample, split them into multiple samples.
- **Annotations:**
  - entity-relation triples

### Supported Tasks and Leaderboards

NLG

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   | non-cat slot span(dialogue act)   |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       62659 |        62659 |         1 |        18.85 |             1 | -                       | -                      | -                              | -                                 |
| validation |        6980 |         6980 |         1 |        21.22 |             1 | -                       | -                      | -                              | -                                 |
| test       |       12552 |        12552 |         1 |        20.95 |             1 | -                       | -                      | -                              | -                                 |
| all        |       82191 |        82191 |         1 |        19.37 |             1 | -                       | -                      | -                              | -                                 |

6 domains: ['WikiTableQuestions_mturk', 'WikiSQL_decl_sents', 'WikiSQL_lily', 'WikiTableQuestions_lily', 'webnlg', 'e2e']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.


### Citation

```
@inproceedings{nan-etal-2021-dart,
    title = "{DART}: Open-Domain Structured Data Record to Text Generation",
    author = "Linyong Nan and Dragomir Radev and Rui Zhang and Amrit Rau and Abhinand Sivaprasad and Chiachun Hsieh and Xiangru Tang and Aadit Vyas and Neha Verma and Pranav Krishna and Yangxiaokang Liu and Nadia Irwanto and Jessica Pan and Faiaz Rahman and Ahmad Zaidi and Murori Mutuma and Yasin Tarabar and Ankit Gupta and Tao Yu and Yi Chern Tan and Xi Victoria Lin and Caiming Xiong and Richard Socher and Nazneen Fatema Rajani",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.37",
}

```

### Licensing Information

MIT License