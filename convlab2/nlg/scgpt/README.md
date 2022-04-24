# SC-GPT

This is the implemention of [SC-GPT](https://aclanthology.org/2020.findings-emnlp.17) which is proposed by
Peng et al., 2020.
You should first download and unzip the SG-GPT checkpoint
```bash
wget https://bapengstorage.blob.core.windows.net/fileshare/scgpt.tar.gz
tar -xvf scgpt.tar.gz
```
and if you want to use this checkpoint, you have to specifiy its path through ``--scgpt_model_ckpt_path`` parameter in ``train.sh`` and ``test.sh``.

## Train

```python
./train.sh
```
When using the training code, you may have to adjust the parameters 
according to your machine configuration. Note that the training code
only supports GPU training.

## Evaluation
```python
./evaluate.sh
```
The test code also only supports GPU mode. We will report the BLEU score
and ERR score according to the original SC-GPT paper(Peng et al., 2020).

## NLG Interface
The NLG interface of SC-GPT is implemented in ./scgpt.py.
```python
def generate(self, action)
```
This class supports both CPU and GPU mode by providing the
'device' parameter in constructor function.


## Reference
```
@inproceedings{peng-etal-2020-shot,
    title = "Few-shot Natural Language Generation for Task-Oriented Dialog",
    author = "Peng, Baolin and Zhu, Chenguang  and  Li, Chunyuan  and  Li, Xiujun  and  Li, Jinchao  and  Zeng, Michael  and  Gao, Jianfeng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "172--182",
}
```