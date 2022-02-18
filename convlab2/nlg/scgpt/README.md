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
python train.py --output_dir=trained_output --model_type=gpt2 --model_name_or_path=scgpt --do_train --do_eval --eval_data_file=multiwoz/data/test_sys.txt --use_tokenize --train_data_file=multiwoz/data/train_sys.txt --overwrite_output_dir
```

some tricks (optional training argument):
* `--gradient_accumulation_steps xxx` 
* `--fp16`, if it's set, you'd better set `--per_gpu_train_batch_size` to be multiple of 8
* `--max_seq xxx`, it should be larger than the length of the longest sequence. You can set `--max_seq 1024`. The script uses a dynamic sequence length at each training step.
* `--gradient_checkpointing`, it allows larger `per_gpu_train_batch_size`
* `--use_multi_tensor_adamw`, someone says it's a faster optimizer

distributed data parallel:

  If multiple GPUs are available, you can run `python -m torch.distributed.launch --nproc_per_node CUDA_COUNT train.py ......` 

  `CUDA_COUNT` is the number of GPUs. `.....` are arguments of `train.py`.

## Use

```python
python run.py --model_type=gpt2 --model_name_or_path=$save_dir$ --num_samples 5 --input_file=$test_file$ --output_file=$output_file$ --length 100 --stop_token '<|endoftext|>' --batch_size 16
```

## Data Format

```
dialog act seq & user utterance
```

