# MILU
MILU is a joint neural model that allows you to simultaneously predict multiple dialog act items (a dialog act item takes a form of domain-intent(slot, value). Since it is common that, in a multi-domain setting, an utterance has multiple dialog act items, MILU is likely to yield higher performance than conventional single-intent models.


## Example usage
We based our implementation on the [AllenNLP library](https://github.com/allenai/allennlp). For an introduction to this library, you should check [these tutorials](https://allennlp.org/tutorials).

To use this model, you need to additionally install `overrides==4.1.2, allennlp==0.9.0` and use `python>=3.6,<=3.8`.

### On MultiWOZ dataset

```bash
$ python train.py multiwoz/configs/[base|context3].jsonnet -s serialization_dir
$ python evaluate.py serialization_dir/model.tar.gz {test_file} --cuda-device {CUDA_DEVICE}
```

If you want to perform end-to-end evaluation, you can include the trained model by adding the model path (serialization_dir/model.tar.gz) to your ConvLab spec file.

#### Data
We use the multiwoz data (data/multiwoz/[train|val|test].json.zip).

### MILU on datasets in unified format
We support training MILU on datasets that are in our unified format.

- For **non-categorical** dialogue acts whose values are in the utterances, we use **slot tagging** to extract the values.
- For **categorical** and **binary** dialogue acts whose values may not be presented in the utterances, we treat them as **intents** of the utterances.

Takes MultiWOZ 2.1 (unified format) as an example,
```bash
$ python train.py unified_datasets/configs/multiwoz21_user_context3.jsonnet -s serialization_dir
$ python evaluate.py serialization_dir/model.tar.gz test --cuda-device {CUDA_DEVICE} --output_file output/multiwoz21_user/output.json

# to generate output/multiwoz21_user/predictions.json that merges test data and model predictions.
$ python unified_datasets/merge_predict_res.py -d multiwoz21 -s user -p output/multiwoz21_user/output.json
```
Note that the config file is different from the above. You should set:
- `"use_unified_datasets": true` in `dataset_reader` and `model`
- `"dataset_name": "multiwoz21"` in `dataset_reader`
- `"train_data_path": "train"`
- `"validation_data_path": "validation"`
- `"test_data_path": "test"`

## Predict
See `nlu.py` under `multiwoz` and `unified_datasets` directories.

## References
```
@inproceedings{lee2019convlab,
  title={ConvLab: Multi-Domain End-to-End Dialog System Platform},
  author={Lee, Sungjin and Zhu, Qi and Takanobu, Ryuichi and Li, Xiang and Zhang, Yaoqin and Zhang, Zheng and Li, Jinchao and Peng, Baolin and Li, Xiujun and Huang, Minlie and Gao, Jianfeng},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```
