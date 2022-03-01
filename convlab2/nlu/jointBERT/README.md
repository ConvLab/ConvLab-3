# BERTNLU

On top of the pre-trained BERT, BERTNLU use an MLP for slot tagging and another MLP for intent classification. All parameters are fine-tuned to learn these two tasks jointly.

Dialog acts are split into two groups, depending on whether the values are in the utterances:

- For dialogue acts whose values are in the utterances, we use **slot tagging** to extract the values. For example, `"Find me a cheap hotel"`, its dialog act is `{intent=Inform, domain=hotel, slot=price, value=cheap}`, and the corresponding BIO tag sequence is `["O", "O", "O", "B-inform-hotel-price", "O"]`. An MLP classifier takes a token's representation from BERT and outputs its tag.
- For dialogue acts whose values may not be presented in the utterances, we treat them as **intents** of the utterances. Another MLP takes embeddings of `[CLS]` of a utterance as input and does the binary classification for each intent independently. Since some intents are rare, we set the weight of positive samples as $\lg(\frac{\# \ negative\ samples}{\# \ positive\ samples})$ empirically for each intent.

The model can also incorporate context information by setting the `context=true` in the config file. The context utterances will be concatenated (separated by `[SEP]`) and fed into BERT. Then the `[CLS]` embedding serves as context representaion and is concatenated to all token representations in the target utterance right before the slot and intent classifiers.


## Usage

Follow the instruction under each dataset's directory to prepare data and model config file for training and evaluation.

#### Train a model

```sh
$ python train.py --config_path path_to_a_config_file
```

The model (`pytorch_model.bin`) will be saved under the `output_dir` of the config file.

#### Test a model

```sh
$ python test.py --config_path path_to_a_config_file
```

The result (`output.json`) will be saved under the `output_dir` of the config file. Also, it will be zipped as `zipped_model_path` in the config file.


## References

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}

@inproceedings{zhu-etal-2020-convlab,
    title = "{C}onv{L}ab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems",
    author = "Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-demos.19",
    doi = "10.18653/v1/2020.acl-demos.19",
    pages = "142--149"
}
```