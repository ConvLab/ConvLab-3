# Emotion Recogniser - ContextBERT-ERToD
ContextBERT-ERToD is an emotion classifier dedicated to task-oriented scenario. It is based on ContextBERT used to benchmark EmoWOZ dataset. In addition to dialogue history, it considers dialogue state as additional input features, uses augmented data for rare emotions, incorporates a distance-based emotion loss function, incorporates multi-task learning, and is initialised with a sentiment-aware version of BERT. 

## Environment
We recommend the environment specified in the [EmoWOZ repository](https://gitlab.cs.uni-duesseldorf.de/general/dsml/emowoz-public) (Feng et al., 2022).

## Data and Pre-trained model checkpoint

Please obtain the pre-trained sentiment-aware embedding from [here](https://github.com/DrJZhou/SentiX?tab=readme-ov-file). Alternatively, the vanilla BERT and its other derivatives are also supported. You can specify the initialisation with --pretrained_model_dir argument.

Please obtain the data from the [here]().

## Training and Testing

```
python train_contextbert_ertod.py \
    --exp_id ${path_to_the_folder_for_saving_trained_model} \
    --seed ${seed} \
    --epochs 5 \
    --data_dir ${path_to_the_data_folder} \
    --pretrained_model_dir ${path_to_the_pretrained_model_file} \
    --model_checkpoint \ # specify this to test a trained model only
    --use_context \
    --emotion \
    --augment fearful apologetic abusive excited \
    --augment_src to-inferred \
    --dialog_state \
    --valence \
    --elicitor \
    --conduct \
    --distance_loss
```

## References

```
@inproceedings{feng-etal-2022-emowoz,
    title = "{E}mo{WOZ}: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems",
    author = "Feng, Shutong  and
      Lubis, Nurul  and
      Geishauser, Christian  and
      Lin, Hsien-chin  and
      Heck, Michael  and
      van Niekerk, Carel  and
      Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.436",
    pages = "4096--4113",
}
```

```
@inproceedings{feng-etal-2023-chatter,
    title = "From Chatter to Matter: Addressing Critical Steps of Emotion Recognition Learning in Task-oriented Dialogue",
    author = "Feng, Shutong  and
      Lubis, Nurul  and
      Ruppik, Benjamin  and
      Geishauser, Christian  and
      Heck, Michael  and
      Lin, Hsien-chin  and
      van Niekerk, Carel  and
      Vukovic, Renato  and
      Gasic, Milica",
    editor = "Stoyanchev, Svetlana  and
      Joty, Shafiq  and
      Schlangen, David  and
      Dusek, Ondrej  and
      Kennington, Casey  and
      Alikhani, Malihe",
    booktitle = "Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2023",
    address = "Prague, Czechia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sigdial-1.8",
    doi = "10.18653/v1/2023.sigdial-1.8",
    pages = "85--103",
    abstract = "Emotion recognition in conversations (ERC) is a crucial task for building human-like conversational agents. While substantial efforts have been devoted to ERC for chit-chat dialogues, the task-oriented counterpart is largely left unattended. Directly applying chit-chat ERC models to task-oriented dialogues (ToDs) results in suboptimal performance as these models overlook key features such as the correlation between emotions and task completion in ToDs. In this paper, we propose a framework that turns a chit-chat ERC model into a task-oriented one, addressing three critical aspects: data, features and objective. First, we devise two ways of augmenting rare emotions to improve ERC performance. Second, we use dialogue states as auxiliary features to incorporate key information from the goal of the user. Lastly, we leverage a multi-aspect emotion definition in ToDs to devise a multi-task learning objective and a novel emotion-distance weighted loss function. Our framework yields significant improvements for a range of chit-chat ERC models on EmoWOZ, a large-scale dataset for user emotions in ToDs. We further investigate the generalisability of the best resulting model to predict user satisfaction in different ToD datasets. A comparison with supervised baselines shows a strong zero-shot capability, highlighting the potential usage of our framework in wider scenarios.",
}
```