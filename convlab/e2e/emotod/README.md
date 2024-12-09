# EmoLLAMA as used in the work of EmoLoop

This is the repository for the llama2-based end-to-end system used in the work of EmoLoop (Feng et al., 2024). This model is based on the work of Stricker and Paroubek (2024).

## Model Preparation and Training

The model can be trained from [this repository](https://github.com/armandstrickernlp/Emo-TOD). Simply add `<|conduct|> {SYSTEM_CONDUCT} <|endofconduct|>` when formatting the training data after dialogue action and `<|conduct|>` and `<|endofconduct|>` as additional special vocabularies to the tokenizer. System conduct labels come from [EmoWOZ 2.0](). After training, you can specify the saved model path to initialise the `EMOLLAMAAgent` object in `emollama.py`.

To obtain SimpleLLAMA, follow the `simple` set-up in the respository to have a non-emotional system.

## Evaluation

The corpus evaluation metrics will be printed out and logged after training using the repository mentioned above. To evaluate the system with the user simulator, run the following code
```
python run_interaction.py \
    --model_path path_to_the_emollama_state_dict \
    --output_path path_to_the_folder_where_dialogues_and_results_are_saved \
    --emous_path path_to_langEmoUS_checkpoint \
    --user_nlu_path path_to_user_nlu \ we used t5-small-nlu-all-multiwoz21-context3, which can be found in Huggingface ConvLab repository.
    --num_dialogues number_of_dialogues_to_simulate \
    --seed random_seed \
    --simple # specify this if the end-to-end system checkpoint is SimpleLLAMA
    
```
*For this script, we recommend simulating a small number of dialogues for multiple times with different random seed because there may be unexpected errors during the lexicalisation process and model output parsing.

## Reference

```
@misc{stricker2024unifiedapproachemotiondetection,
      title={A Unified Approach to Emotion Detection and Task-Oriented Dialogue Modeling}, 
      author={Armand Stricker and Patrick Paroubek},
      year={2024},
      eprint={2401.13789},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.13789}, 
}

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