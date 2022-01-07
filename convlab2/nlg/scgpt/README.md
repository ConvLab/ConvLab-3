# GPT

The code derives from [HuggingFace/Transformers](https://github.com/huggingface/transformers).

## Preprocess

```python
cd $dataset$
python preprocess.py
```

## Train

Fetch and unzip the checkpoint

```
wget https://bapengstorage.blob.core.windows.net/fileshare/scgpt.tar.gz
tar -xvf scgpt.tar.gz
```

Then

``` python
python train.py --output_dir=trained_output --model_type=gpt2 --model_name_or_path=scgpt --do_train --do_eval --eval_data_file=multiwoz/data/test_sys.txt --overwrite_cache --use_tokenize --train_data_file=multiwoz/data/train_sys.txt --overwrite_output_dir
```

## Use

```python
python run.py --model_type=gpt2 --model_name_or_path=$save_dir$ --num_samples 5 --input_file=$test_file$ --output_file=$output_file$ --length 100 --stop_token '<|endoftext|>' --batch_size 16
```

## Data Format

```
dialog act seq & user utterance
```

