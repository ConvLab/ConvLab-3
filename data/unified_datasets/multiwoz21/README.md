# Dataset Card for MultiWOZ 2.1

- **Repository:** https://github.com/budzianowski/multiwoz
- **Paper:** https://aclanthology.org/2020.lrec-1.53
- **Leaderboard:** https://github.com/budzianowski/multiwoz
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

MultiWOZ 2.1 fixed the noise in state annotations and dialogue utterances. It also includes user dialogue acts from ConvLab (Lee et al., 2019) as well as multiple slot descriptions per dialogue state slot.

- **How to get the transformed data from original data:** 
  - Download [MultiWOZ_2.1.zip](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip).
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Create a new ontology in the unified format, taking slot descriptions from MultiWOZ 2.2.
  - Correct some grammar errors in the text, mainly following `tokenization.md` in MultiWOZ_2.1.
  - Normalize slot name and value. See `normalize_domain_slot_value` function in `preprocess.py`.
  - Correct some non-categorical slots' values and provide character level span annotation.
  - Concatenate multiple values in user goal & state using `|`.
  - Add `booked` information in system turns from original belief states.
  - Remove `Booking` domain and remap all booking relevant dialog acts to unify the annotation of booking action in different domains, see `booking_remapper.py`.
- **Annotations:**
  - user goal, dialogue acts, state.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG, E2E, User simulator

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains |   cat slot match(state) |   cat slot match(goal) |   cat slot match(dialogue act) |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |        8438 |       113556 |     13.46 |        13.23 |          2.8  |                   98.84 |                  99.48 |                          86.39 |                             98.22 |
| validation |        1000 |        14748 |     14.75 |        13.5  |          2.98 |                   98.84 |                  99.46 |                          86.59 |                             98.17 |
| test       |        1000 |        14744 |     14.74 |        13.5  |          2.93 |                   99.21 |                  99.32 |                          85.83 |                             98.58 |
| all        |       10438 |       143048 |     13.7  |        13.28 |          2.83 |                   98.88 |                  99.47 |                          86.35 |                             98.25 |

8 domains: ['attraction', 'hotel', 'taxi', 'restaurant', 'train', 'police', 'hospital', 'general']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{eric-etal-2020-multiwoz,
    title = "{M}ulti{WOZ} 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines",
    author = "Eric, Mihail and Goel, Rahul and Paul, Shachi and Sethi, Abhishek and Agarwal, Sanchit and Gao, Shuyag and Hakkani-Tur, Dilek",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.53",
    pages = "422--428",
    ISBN = "979-10-95546-34-4",
}
```

### Licensing Information

Apache License, Version 2.0