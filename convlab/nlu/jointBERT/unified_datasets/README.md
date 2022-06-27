# BERTNLU on datasets in unified format

We support training BERTNLU on datasets that are in our unified format.

- For **non-categorical** dialogue acts whose values are in the utterances, we use **slot tagging** to extract the values.
- For **categorical** and **binary** dialogue acts whose values may not be presented in the utterances, we treat them as **intents** of the utterances.

## Usage

#### Preprocess data

```sh
$ python preprocess.py --dataset dataset_name --speaker {user,system,all} --context_window_size CONTEXT_WINDOW_SIZE --save_dir save_directory
```

Note that the dataset will be loaded by `convlab.util.load_dataset(dataset_name)`. If you want to use custom datasets, make sure they follow the unified format and can be loaded using this function.
output processed data on `${save_dir}/${dataset_name}/${speaker}/context_window_size_${context_window_size}` dir.

#### Train a model

Prepare a config file and run the training script in the parent directory:

```sh
$ python train.py --config_path path_to_a_config_file
```

The model (`pytorch_model.bin`) will be saved under the `output_dir` of the config file. Also, it will be zipped as `zipped_model_path` in the config file.

#### Test a model

Run the inference script in the parent directory:

```sh
$ python test.py --config_path path_to_a_config_file
```

The result (`output.json`) will be saved under the `output_dir` of the config file.

To generate `predictions.json` that merges test data and model predictions under the same directory of the `output.json`:
```sh
$ python merge_predict_res.py -d dataset_name -s {user,system,all} -c CONTEXT_WINDOW_SIZE -p path_to_output.json
```

#### Predict

See `nlu.py` for usage.
