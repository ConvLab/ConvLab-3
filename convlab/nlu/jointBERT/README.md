# BERTNLU

On top of the pre-trained BERT, BERTNLU use an MLP for slot tagging and another MLP for intent classification. All parameters are fine-tuned to learn these two tasks jointly.

Dialog acts are split into two groups, depending on whether the values are in the utterances:

- For dialogue acts whose values are in the utterances, we use **slot tagging** to extract the values. For example, `"Find me a cheap hotel"`, its dialog act is `{intent=Inform, domain=hotel, slot=price, value=cheap}`, and the corresponding BIO tag sequence is `["O", "O", "O", "B-inform-hotel-price", "O"]`. An MLP classifier takes a token's representation from BERT and outputs its tag.
- For dialogue acts whose values may not be presented in the utterances, we treat them as **intents** of the utterances. Another MLP takes embeddings of `[CLS]` of a utterance as input and does the binary classification for each intent independently. Since some intents are rare, we set the weight of positive samples as $\lg(\frac{num\_negative\_samples}{num\_positive\_samples})$ empirically for each intent.

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


## Performance on unified format datasets

To illustrate that it is easy to use the model for any dataset that in our unified format, we report the performance on several datasets in our unified format. We follow `README.md` and config files in `unified_datasets/` to generate `predictions.json`, then evaluate it using `../evaluate_unified_datasets.py`. Note that we use almost the same hyper-parameters for different datasets, which may not be optimal. Trained models are available at [Hugging Face Hub](https://huggingface.co/ConvLab/bert-base-nlu).

<table>
<thead>
  <tr>
    <th></th>
    <th colspan=2>MultiWOZ 2.1</th>
    <th colspan=2>MultiWOZ 2.1 all utterances</th>
    <th colspan=2>Taskmaster-1</th>
    <th colspan=2>Taskmaster-2</th>
    <th colspan=2>Taskmaster-3</th>
  </tr>
</thead>
<thead>
  <tr>
    <th>Model</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
    <th>Acc</th><th>F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BERTNLU</td>
    <td>74.5</td><td>85.9</td>
    <td>59.5</td><td>80.0</td>
    <td>72.8</td><td>50.6</td>
    <td>79.2</td><td>70.6</td>
    <td>86.1</td><td>81.9</td>
  </tr>
  <tr>
    <td>BERTNLU (context=3)</td>
    <td>80.6</td><td>90.3</td>
    <td>58.1</td><td>79.6</td>
    <td>74.2</td><td>52.7</td>
    <td>80.9</td><td>73.3</td>
    <td>87.8</td><td>83.8</td>
  </tr>
</tbody>
</table>

- Acc: whether all dialogue acts of an utterance are correctly predicted
- F1: F1 measure of the dialogue act predictions over the corpus.

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