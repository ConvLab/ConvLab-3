---
language:
- en
license:
- cc-by-sa-4.0
multilinguality:
- monolingual
pretty_name: SGD-X v1
size_categories:
- 10K<n<100K
task_categories:
- conversational
---

# Dataset Card for SGD-X v1

- **Repository:** https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/master/sgd_x
- **Paper:** https://arxiv.org/pdf/2110.06800.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

To use this dataset, you need to install [ConvLab-3](https://github.com/ConvLab/ConvLab-3) platform first. Then you can load the dataset via:
```
from convlab.util import load_dataset, load_ontology, load_database

dataset = load_dataset('sgd1')
ontology = load_ontology('sgd1')
database = load_database('sgd1')
```
For more usage please refer to [here](https://github.com/ConvLab/ConvLab-3/tree/master/data/unified_datasets).

### Dataset Summary

The **Schema-Guided Dialogue (SGD)** dataset consists of over 20k annotated multi-domain, task-oriented conversations between a human and a virtual assistant. These conversations involve interactions with services and APIs spanning 20 domains, such as banks, events, media, calendar, travel, and weather. For most of these domains, the dataset contains multiple different APIs, many of which have overlapping functionalities but different interfaces, which reflects common real-world scenarios. The wide range of available annotations can be used for intent prediction, slot filling, dialogue state tracking, policy imitation learning, language generation, and user simulation learning, among other tasks for developing large-scale virtual assistants. Additionally, the dataset contains unseen domains and services in the evaluation set to quantify the performance in zero-shot or few-shot settings.

The **SGD-X** dataset consists of 5 linguistic variants of every schema in the original SGD dataset. Linguistic variants were written by hundreds of paid crowd-workers. In the SGD-X directory, v1 represents the variant closest to the original schemas and v5 the farthest in terms of linguistic distance. To evaluate model performance on SGD-X schemas, dialogues must be converted using the script generate_sgdx_dialogues.py.

- **How to get the transformed data from original data:** 
  - Download [dstc8-schema-guided-dialogue-master.zip](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip).
  - Modified `sgd_x/generate_sgdx_dialogues.py` as https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/issues/57
  - Run `python -m sgd_x.generate_sgdx_dialogues` under `dstc8-schema-guided-dialogue-master` dir which need tensorflow installed.
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Lower case original `act` as `intent`.
  - Add `count` slot for each domain, non-categorical, find span by text matching.
  - Categorize `dialogue acts` according to the `intent`.
  - Concatenate multiple values using `|`.
  - Retain `active_intent`, `requested_slots`, `service_call`.
- **Annotations:**
  - dialogue acts, state, db_results, service_call, active_intent, requested_slots.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG, E2E

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains |   cat slot match(state) | cat slot match(goal)   |   cat slot match(dialogue act) |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       16142 |       329964 |     20.44 |         9.75 |          1.84 |                     100 | -                      |                            100 |                               100 |
| validation |        2482 |        48726 |     19.63 |         9.66 |          1.84 |                     100 | -                      |                            100 |                               100 |
| test       |        4201 |        84594 |     20.14 |        10.4  |          2.02 |                     100 | -                      |                            100 |                               100 |
| all        |       22825 |       463284 |     20.3  |         9.86 |          1.87 |                     100 | -                      |                            100 |                               100 |

45 domains: ['Banks_11', 'Buses_11', 'Buses_21', 'Calendar_11', 'Events_11', 'Events_21', 'Flights_11', 'Flights_21', 'Homes_11', 'Hotels_11', 'Hotels_21', 'Hotels_31', 'Media_11', 'Movies_11', 'Music_11', 'Music_21', 'RentalCars_11', 'RentalCars_21', 'Restaurants_11', 'RideSharing_11', 'RideSharing_21', 'Services_11', 'Services_21', 'Services_31', 'Travel_11', 'Weather_11', 'Alarm_11', 'Banks_21', 'Flights_31', 'Hotels_41', 'Media_21', 'Movies_21', 'Restaurants_21', 'Services_41', 'Buses_31', 'Events_31', 'Flights_41', 'Homes_21', 'Media_31', 'Messaging_11', 'Movies_31', 'Music_31', 'Payment_11', 'RentalCars_31', 'Trains_11']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{lee2022sgd,
  title={SGD-X: A Benchmark for Robust Generalization in Schema-Guided Dialogue Systems},
  author={Lee, Harrison and Gupta, Raghav and Rastogi, Abhinav and Cao, Yuan and Zhang, Bin and Wu, Yonghui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={10938--10946},
  year={2022}
}
```

### Licensing Information

[**CC BY-SA 4.0**](https://creativecommons.org/licenses/by-sa/4.0/)