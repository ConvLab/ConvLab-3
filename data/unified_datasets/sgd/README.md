# Dataset Card for Schema-Guided Dialogue

- **Repository:** https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- **Paper:** https://arxiv.org/pdf/1909.05855.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

The **Schema-Guided Dialogue (SGD)** dataset consists of over 20k annotated multi-domain, task-oriented conversations between a human and a virtual assistant. These conversations involve interactions with services and APIs spanning 20 domains, such as banks, events, media, calendar, travel, and weather. For most of these domains, the dataset contains multiple different APIs, many of which have overlapping functionalities but different interfaces, which reflects common real-world scenarios. The wide range of available annotations can be used for intent prediction, slot filling, dialogue state tracking, policy imitation learning, language generation, and user simulation learning, among other tasks for developing large-scale virtual assistants. Additionally, the dataset contains unseen domains and services in the evaluation set to quantify the performance in zero-shot or few-shot settings.

- **How to get the transformed data from original data:** 
  - Download [dstc8-schema-guided-dialogue-master.zip](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip).
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

| split      | dialogues | utterances | avg_utt | avg_tokens | avg_domains | cat slot match(state) | cat slot match(goal) | cat slot match(dialogue act) | non-cat slot span(dialogue act) |
| ---------- | --------- | ---------- | ------- | ---------- | ----------- | --------------------- | -------------------- | ---------------------------- | ------------------------------- |
| train      | 16142     | 329964     | 20.44   | 9.75       | 1.84        | 100                   | -                    | 100                          | 100                             |
| validation | 2482      | 48726      | 19.63   | 9.66       | 1.84        | 100                   | -                    | 100                          | 100                             |
| test       | 4201      | 84594      | 20.14   | 10.4       | 2.02        | 100                   | -                    | 100                          | 100                             |
| all        | 22825     | 463284     | 20.3    | 9.86       | 1.87        | 100                   | -                    | 100                          | 100                             |

45 domains: ['Banks_1', 'Buses_1', 'Buses_2', 'Calendar_1', 'Events_1', 'Events_2', 'Flights_1', 'Flights_2', 'Homes_1', 'Hotels_1', 'Hotels_2', 'Hotels_3', 'Media_1', 'Movies_1', 'Music_1', 'Music_2', 'RentalCars_1', 'RentalCars_2', 'Restaurants_1', 'RideSharing_1', 'RideSharing_2', 'Services_1', 'Services_2', 'Services_3', 'Travel_1', 'Weather_1', 'Alarm_1', 'Banks_2', 'Flights_3', 'Hotels_4', 'Media_2', 'Movies_2', 'Restaurants_2', 'Services_4', 'Buses_3', 'Events_3', 'Flights_4', 'Homes_2', 'Media_3', 'Messaging_1', 'Movies_3', 'Music_3', 'Payment_1', 'RentalCars_3', 'Trains_1']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@article{rastogi2019towards,
  title={Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset},
  author={Rastogi, Abhinav and Zang, Xiaoxue and Sunkara, Srinivas and Gupta, Raghav and Khaitan, Pranav},
  journal={arXiv preprint arXiv:1909.05855},
  year={2019}
}
```

### Licensing Information

[**CC BY-SA 4.0**](https://creativecommons.org/licenses/by-sa/4.0/)