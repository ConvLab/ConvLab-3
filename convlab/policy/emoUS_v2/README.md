# langEmoUS

This is the user simulator used to train the emotion-aware policy in EmoLoop (Feng et al., 2024) on the natural language level. It is built on the semantic-level emotional user simulator EmoUS (Lin et al., 2023).

## Data Preparation

In order to train langEmoUS from scratch, please download the data from [here](), unzip the file, and place the `data` folder in `convlab/policy/emoUS_v2/unify`. Then run the following command *from the base convlab directory*.

```
python convlab/policy/emoUS_v2/unify/build_data.py --add-persona --language --dataset emowoz+dialmage
```

## Running langEmoUS

To train the model, running the following command *from the base convlab directory*.
```
python convlab/policy/emoUS_v2/train_model.py --data-name language_EmoUS_emowoz+dialmage --batch-size 4 --max-in-len 800 --max-out-len 400
```

To interact with langEmoUS and verify that if works, please download the langEmoUS model checkpoint from [here]() and run the following command *from the base convlab directory*.

```
python convlab/policy/emoUS_v2/evaluate.py \
    --model-checkpoint $path_to_the_model_checkpoint \
    --input-file $path_to_the_test_data \ # e.g. convlab/policy/emoUS_v2/unify/data/language_EmoUS_emowoz+dialmage_0_1/test.json
    --Neutral 0.95 \
    --Fearful 1 \
    --Dissatisfied 1 \
    --Apologetic 1 \
    --Abusive 1 \
    --Excited 1 \
    --Satisfied 0.95 \
    --result-base-name $name_of_result_folder \
    --language # set this flag to evaluate langEmoUS on the language level
```

## References

```
@inproceedings{10.1145/3539618.3592092,
author = {Lin, Hsien-Chin and Feng, Shutong and Geishauser, Christian and Lubis, Nurul and van Niekerk, Carel and Heck, Michael and Ruppik, Benjamin and Vukovic, Renato and Gasi\'{c}, Milica},
title = {EmoUS: Simulating User Emotions in Task-Oriented Dialogues},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3592092},
doi = {10.1145/3539618.3592092},
abstract = {Existing user simulators (USs) for task-oriented dialogue systems only model user behaviour on semantic and natural language levels without considering the user persona and emotions. Optimising dialogue systems with generic user policies, which cannot model diverse user behaviour driven by different emotional states, may result in a high drop-off rate when deployed in the real world. Thus, we present EmoUS, a user simulator that learns to simulate user emotions alongside user behaviour. EmoUS generates user emotions, semantic actions, and natural language responses based on the user goal, the dialogue history, and the user persona. By analysing what kind of system behaviour elicits what kind of user emotions, we show that EmoUS can be used as a probe to evaluate a variety of dialogue systems and in particular their effect on the user's emotional state. Developing such methods is important in the age of large language model chat-bots and rising ethical concerns.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2526–2531},
numpages = {6},
keywords = {dialogue system, emotion simulation, user simulation},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```

```
@inproceedings{feng-etal-2024-infusing,
    title = "Infusing Emotions into Task-oriented Dialogue Systems: Understanding, Management, and Generation",
    author = "Feng, Shutong  and
      Lin, Hsien-chin  and
      Geishauser, Christian  and
      Lubis, Nurul  and
      van Niekerk, Carel  and
      Heck, Michael  and
      Ruppik, Benjamin Matthias  and
      Vukovic, Renato  and
      Gasic, Milica",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.60",
    doi = "10.18653/v1/2024.sigdial-1.60",
    pages = "699--717",
    abstract = "Emotions are indispensable in human communication, but are often overlooked in task-oriented dialogue (ToD) modelling, where the task success is the primary focus. While existing works have explored user emotions or similar concepts in some ToD tasks, none has so far included emotion modelling into a fully-fledged ToD system nor conducted interaction with human or simulated users. In this work, we incorporate emotion into the complete ToD processing loop, involving understanding, management, and generation. To this end, we extend the EmoWOZ dataset (Feng et al., 2022) with system affective behaviour labels. Through interactive experimentation involving both simulated and human users, we demonstrate that our proposed framework significantly enhances the user{'}s emotional experience as well as the task success.",
}
```