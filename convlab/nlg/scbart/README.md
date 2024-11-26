# SEC-BART

This is the NLG used in EmoLoop system (Feng et al., 2024). EmoLoop is trained on EmoWOZ 2.0, adding additional system conduct labels for original [EmoWOZ](https://zenodo.org/records/6506504) (Feng et al., 2022). EmoWOZ 2.0 can be found [here]().

## Data

Please download the data from [here]().

## Training

```
python train.py \
    --exp_id ${path_to_the_folder_for_saving_trained_model} \
    --data_dir ${path_to_the_data_folder} \ # without the last slash
    --emowoz2_prev_user_utt # to specify the template format

```

To test the NLG, run `python test_nlg.py` after specifying the `model_path` argument in the script.

## Evaluation
The training script will store prediction results on the test set in a csv file named `{args.exp_id}/test-best-temperate{args.temperature}.csv`. To evaluate the result in the ConvLab-3 environment, you need to first convert it to the ConvLab-3 unified data format for nlg task (the data object returned from `convlab.util.unified_datasets_util.load_nlg_data` with an additional `prediction` field in each turn). To do so, run the following command:

```
python convert_result_to_unified.py \
  --predict_result {path_to_the_csv_file_containing_inference_result} \
  --emowoz2 {path_to_the_json_file_containing_system_conduct_labels}
  --output {path_to_the_json_file_storing_converted_data} # default scbart_predict_result_unified.json
```

Then, use the script in convlab/nlg/evaluate_unified_datasets_v2.py to obtain evaluation metrics.

```
python evaluate_unified_datasets_v2.py \
  --predict_result {path_to_the_json_file_storing_converted_data} \ # e.g. scbart/scbart_predict_result_unified.json
  --dataset_name multiwoz21
```

The script should print a dictionary of metrics looking something like
```
{'bleu': 36.87063725514231,
 'err': 0.0371580264616651,
 'missing': 282,
 'redundant': 157,
 'total': 11799}
```
where 'err' means the slot error rate: err = (missing + redundant) / total. 

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