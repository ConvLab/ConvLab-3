# SC-GPT

This is the implemention of [SC-GPT](https://aclanthology.org/2020.findings-emnlp.17) which is proposed by
Peng et al., 2020.

## Train

```python
./train.sh
```
When using the training code, you may have to adjust the parameters 
according to your machine configuration. Note that the training code
only supports GPU training.

## Test
```python
./test.sh
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
