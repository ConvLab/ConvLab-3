# Dataset Card for Taskmaster-1

- **Repository:** https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020
- **Paper:** https://aclanthology.org/2021.acl-long.55.pdf
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

The Taskmaster-3 (aka TicketTalk) dataset consists of 23,789 movie ticketing dialogs (located in Taskmaster/TM-3-2020/data/). By "movie ticketing" we mean conversations where the customer's goal is to purchase tickets after deciding on theater, time, movie name, number of tickets, and date, or opt out of the transaction.

This collection was created using the "self-dialog" method. This means a single, crowd-sourced worker is paid to create a conversation writing turns for both speakers, i.e. the customer and the ticketing agent. In order to gather a wide range of conversational scenarios and linguistic phenomena, workers were given both open-ended as well as highly structured conversational tasks. In all, we used over three dozen sets of instructions while building this corpus. The "instructions" field in data.json provides the exact scenario workers were given to complete each dialog. In this way, conversations involve a wide variety of paths, from those where the customer decides on a movie based on genre, their location, current releases, or from what they already have in mind. In addition, dialogs also include error handling with repect to repair (e.g. "No, I said Tom Cruise."), clarifications (e.g. "Sorry. Did you want the AMC 16 or Century City 16?") and other common conversational hiccups. In some cases instructions are completely open ended e.g. "Pretend you are taking your friend to a movie in Salem, Oregon. Create a conversation where you end up buying two tickets after finding out what is playing in at least two local theaters. Make sure the ticket purchase includes a confirmation of the deatils by the agent before the purchase, including date, time, movie, theater, and number of tickets." In other cases we restrict the conversational content and structure by offering a partially completed conversation that the workers must finalize or fill in based a certain parameters. These partially completed dialogs are labeled "Auto template" in the "scenario" field shown for each conversation in the data.json file. In some cases, we provided a small KB from which workers would choose movies, theaters, etc. but in most cases (pre-pandemic) workers were told to use the internet to get accurate current details for their dialogs. In any case, all relevant entities are annotated.

- **How to get the transformed data from original data:** 
  - Download [master.zip](https://github.com/google-research-datasets/Taskmaster/archive/refs/heads/master.zip).
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - Remove dialogs that are empty or only contain one speaker.
  - Split each domain dialogs into train/validation/test randomly (8:1:1).
  - Merge continuous turns by the same speaker (ignore repeated turns).
  - Annotate `dialogue acts` according to the original segment annotations. Add `intent` annotation (`==inform`). The type of `dialogue act` is set to `non-categorical` if the `slot` is not `description.other` or `description.plot`. Otherwise, the type is set to `binary` (and the `value` is empty). If there are multiple spans overlapping, we only keep the shortest one, since we found that this simple strategy can reduce the noise in annotation.
  - Add `domain` and `intent` descriptions.
  - Rename `api` to `db_results`.
  - Add `state` by accumulate `non-categorical dialogue acts` in the order that they appear.
- **Annotations:**
  - dialogue acts, state, db_results.

### Supported Tasks and Leaderboards

NLU, DST, Policy, NLG, E2E

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   |   non-cat slot span(dialogue act) |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       18997 |       380646 |     20.04 |        10.48 |             1 | -                       | -                      | -                              |                               100 |
| validation |        2380 |        47531 |     19.97 |        10.38 |             1 | -                       | -                      | -                              |                               100 |
| test       |        2380 |        48849 |     20.52 |        10.12 |             1 | -                       | -                      | -                              |                               100 |
| all        |       23757 |       477026 |     20.08 |        10.43 |             1 | -                       | -                      | -                              |                               100 |

1 domains: ['movie']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{byrne-etal-2021-tickettalk,
    title = "{T}icket{T}alk: Toward human-level performance with end-to-end, transaction-based dialog systems",
    author = "Byrne, Bill  and
      Krishnamoorthi, Karthik  and
      Ganesh, Saravanan  and
      Kale, Mihir",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.55",
    doi = "10.18653/v1/2021.acl-long.55",
    pages = "671--680",
}
```

### Licensing Information

[**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/)