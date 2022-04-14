# Dataset Card for MetaLWOZ

- **Repository:** https://www.microsoft.com/en-us/research/project/metalwoz/
- **Paper:** https://www.microsoft.com/en-us/research/publication/results-of-the-multi-domain-task-completion-dialog-challenge/
- **Leaderboard:** None
- **Who transforms the dataset:** Qi Zhu(zhuq96 at gmail dot com)

### Dataset Summary

This large dataset was created by crowdsourcing 37,884 goal-oriented dialogs, covering 227 tasks in 47 domains. Domains include bus schedules, apartment search, alarm setting, banking, and event reservation. Each dialog was grounded in a scenario with roles, pairing a person acting as the bot and a person acting as the user. (This is the Wizard of Oz reference—using people behind the curtain who act as the machine). Each pair were given a domain and a task, and instructed to converse for 10 turns to satisfy the user’s queries. For example, if a user asked if a bus stop was operational, the bot would respond that the bus stop had been moved two blocks north, which starts a conversation that addresses the user’s actual need.

- **How to get the transformed data from original data:** 
  - Download [metalwoz-v1.zip](https://www.microsoft.com/en-us/download/58389) and [metalwoz-test-v1.zip](https://www.microsoft.com/en-us/download/100639).
  - Run `python preprocess.py` in the current directory.
- **Main changes of the transformation:**
  - `CITI_INFO`, `HOME_BOT`, `NAME_SUGGESTER`, and `TIME_ZONE` are randomly selected as the valiation domains.
  - Remove the first utterance by the system since it is "Hello how may I help you?" in most case.
  - Add goal description according to the original task description: user_role+user_prompt+system_role+system_prompt.
- **Annotations:**
  - domain, goal

### Supported Tasks and Leaderboards

RG, User simulator

### Languages

English

### Data Splits

| split      |   dialogues |   utterances |   avg_utt |   avg_tokens |   avg_domains | cat slot match(state)   | cat slot match(goal)   | cat slot match(dialogue act)   | non-cat slot span(dialogue act)   |
|------------|-------------|--------------|-----------|--------------|---------------|-------------------------|------------------------|--------------------------------|-----------------------------------|
| train      |       34261 |       357092 |     10.42 |         7.48 |             1 | -                       | -                      | -                              | -                                 |
| validation |        3623 |        37060 |     10.23 |         6.59 |             1 | -                       | -                      | -                              | -                                 |
| test       |        2319 |        23882 |     10.3  |         7.96 |             1 | -                       | -                      | -                              | -                                 |
| all        |       40203 |       418034 |     10.4  |         7.43 |             1 | -                       | -                      | -                              | -                                 |

51 domains: ['AGREEMENT_BOT', 'ALARM_SET', 'APARTMENT_FINDER', 'APPOINTMENT_REMINDER', 'AUTO_SORT', 'BANK_BOT', 'BUS_SCHEDULE_BOT', 'CATALOGUE_BOT', 'CHECK_STATUS', 'CITY_INFO', 'CONTACT_MANAGER', 'DECIDER_BOT', 'EDIT_PLAYLIST', 'EVENT_RESERVE', 'GAME_RULES', 'GEOGRAPHY', 'GUINESS_CHECK', 'HOME_BOT', 'HOW_TO_BASIC', 'INSURANCE', 'LIBRARY_REQUEST', 'LOOK_UP_INFO', 'MAKE_RESTAURANT_RESERVATIONS', 'MOVIE_LISTINGS', 'MUSIC_SUGGESTER', 'NAME_SUGGESTER', 'ORDER_PIZZA', 'PET_ADVICE', 'PHONE_PLAN_BOT', 'PHONE_SETTINGS', 'PLAY_TIMES', 'POLICY_BOT', 'PRESENT_IDEAS', 'PROMPT_GENERATOR', 'QUOTE_OF_THE_DAY_BOT', 'RESTAURANT_PICKER', 'SCAM_LOOKUP', 'SHOPPING', 'SKI_BOT', 'SPORTS_INFO', 'STORE_DETAILS', 'TIME_ZONE', 'UPDATE_CALENDAR', 'UPDATE_CONTACT', 'WEATHER_CHECK', 'WEDDING_PLANNER', 'WHAT_IS_IT', 'BOOKING_FLIGHT', 'HOTEL_RESERVE', 'TOURISM', 'VACATION_IDEAS']
- **cat slot match**: how many values of categorical slots are in the possible values of ontology in percentage.
- **non-cat slot span**: how many values of non-categorical slots have span annotation in percentage.

### Citation

```
@inproceedings{li2020results,
    author = {Li, Jinchao and Peng, Baolin and Lee, Sungjin and Gao, Jianfeng and Takanobu, Ryuichi and Zhu, Qi and Minlie Huang and Schulz, Hannes and Atkinson, Adam and Adada, Mahmoud},
    title = {Results of the Multi-Domain Task-Completion Dialog Challenge},
    booktitle = {Proceedings of the 34th AAAI Conference on Artificial Intelligence, Eighth Dialog System Technology Challenge Workshop},
    year = {2020},
    month = {February},
    url = {https://www.microsoft.com/en-us/research/publication/results-of-the-multi-domain-task-completion-dialog-challenge/},
}
```

### Licensing Information

[Microsoft Research Data License Agreement](https://msropendata-web-api.azurewebsites.net/licenses/2f933be3-284d-500b-7ea3-2aa2fd0f1bb2/view)
