# T5 models

By converting NLP tasks into a text-to-text format, we can use one single model to solve various tasks. Here we use T5 as backbone model and provide a unified training script `run_seq2seq.py` for many tasks. **See `*.sh` under each task directory for usage.**

## Create Data
Currently we support natural language understanding (**NLU**), dialog state tracking (**DST**), natural language generation (**NLG**), response generation (**RG**), and generating a dialog from a user goal (**Goal2Dialogue**). We provide serialization and deserialization methods for dialog acts and state in the unified data format (user goals are already natural language instruction). An example of serialized dialog acts and state:

```
User: I am looking for a cheap restaurant.
System: Is there a particular area of town you prefer?
User: In the centre of town.

User dialog acts: [inform][restaurant]([area][centre])
State: [restaurant]([area][centre],[price range][cheap])
System dialog acts: [recommend][restaurant]([name][Zizzi Cambridge])

System: I would recommend Zizzi Cambridge.
```

Dialogue acts are in the form of `[intent][domain]([slot][value],...);...`. State is in the form of `[domain]([slot][value],...);...`. Multiple items will be concatenated by a semicolon `;`.

To create data for a specific task, run `create_data.py` with corresponding arguments. For example, create data for single turn NLU on MultiWOZ 2.1:

```bash
python create_data.py --tasks nlu --datasets multiwoz21 --speaker user
```

Note that the script only supported **datasets in the unified format**.

## Training

To train the model, specify the arguments like data path, learning rate, epochs, etc., and then run `run_seq2seq.py`. See `nlu/run_nlu.sh` for an example.

## Evaluation

The standard evaluation scripts of NLU, DST, and NLG task are located under `../../$task/evaluate_unified_datasets.py` directories. See `nlu/run_nlu.sh` for an example.

## Trained Models

Trained models and their performance are available in [Hugging Face Hub](https://huggingface.co/ConvLab). You can try some example with hosted inference API.

| Name                                                         | Task          | Training Dataset             |
| ------------------------------------------------------------ | ------------- | ---------------------------- |
| [t5-small-goal2dialogue-multiwoz21](https://huggingface.co/ConvLab/t5-small-goal2dialogue-multiwoz21) | Goal2Dialogue | MultiWOZ 2.1                 |
| [t5-small-nlu-multiwoz21](https://huggingface.co/ConvLab/t5-small-nlu-multiwoz21) | NLU           | MultiWOZ 2.1                 |
| [t5-small-nlu-all-multiwoz21](https://huggingface.co/ConvLab/t5-small-nlu-all-multiwoz21) | NLU           | MultiWOZ 2.1 all utterances                |
| [t5-small-nlu-sgd](https://huggingface.co/ConvLab/t5-small-nlu-sgd) | NLU           | SGD                          |
| [t5-small-nlu-tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-nlu-tm1_tm2_tm3) | NLU           | TM1+TM2+TM3                  |
| [t5-small-nlu-multiwoz21_sgd_tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-nlu-multiwoz21_sgd_tm1_tm2_tm3) | NLU           | MultiWOZ 2.1+SGD+TM1+TM2+TM3 |
| [mt5-small-nlu-all-crosswoz](https://huggingface.co/ConvLab/mt5-small-nlu-all-crosswoz) | NLU           | CrossWOZ all utterances                |
| [t5-small-nlu-multiwoz21-context3](https://huggingface.co/ConvLab/t5-small-nlu-multiwoz21-context3) | NLU (context=3)          | MultiWOZ 2.1 |
| [t5-small-nlu-all-multiwoz21-context3](https://huggingface.co/ConvLab/t5-small-nlu-all-multiwoz21-context3) | NLU (context=3)          | MultiWOZ 2.1 all utterances                |
| [t5-small-nlu-tm1-context3](https://huggingface.co/ConvLab/t5-small-nlu-tm1-context3) | NLU (context=3)          | TM1 |
| [t5-small-nlu-tm2-context3](https://huggingface.co/ConvLab/t5-small-nlu-tm2-context3) | NLU (context=3)          | TM2 |
| [t5-small-nlu-tm3-context3](https://huggingface.co/ConvLab/t5-small-nlu-tm3-context3) | NLU (context=3)          | TM3 |
| [t5-small-dst-multiwoz21](https://huggingface.co/ConvLab/t5-small-dst-multiwoz21) | DST           | MultiWOZ 2.1                 |
| [t5-small-dst-sgd](https://huggingface.co/ConvLab/t5-small-dst-sgd) | DST           | SGD                          |
| [t5-small-dst-tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-dst-tm1_tm2_tm3) | DST           | TM1+TM2+TM3                  |
| [mt5-small-dst-crosswoz](https://huggingface.co/ConvLab/mt5-small-dst-crosswoz) | DST           | CrossWOZ                 |
| [t5-small-dst-multiwoz21_sgd_tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-dst-multiwoz21_sgd_tm1_tm2_tm3) | DST           | MultiWOZ 2.1+SGD+TM1+TM2+TM3 |
| [t5-small-nlg-multiwoz21](https://huggingface.co/ConvLab/t5-small-nlg-multiwoz21) | NLG           | MultiWOZ 2.1                 |
| [t5-small-nlg-user-multiwoz21](https://huggingface.co/ConvLab/t5-small-nlg-user-multiwoz21) | NLG           | MultiWOZ 2.1 user utterances                 |
| [t5-small-nlg-all-multiwoz21](https://huggingface.co/ConvLab/t5-small-nlg-all-multiwoz21) | NLG           | MultiWOZ 2.1 all utterances                 |
| [t5-small-nlg-sgd](https://huggingface.co/ConvLab/t5-small-nlg-sgd) | NLG           | SGD                          |
| [t5-small-nlg-tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-nlg-tm1_tm2_tm3) | NLG           | TM1+TM2+TM3                  |
| [t5-small-nlg-multiwoz21_sgd_tm1_tm2_tm3](https://huggingface.co/ConvLab/t5-small-nlg-multiwoz21_sgd_tm1_tm2_tm3) | NLG           | MultiWOZ 2.1+SGD+TM1+TM2+TM3 |
| [mt5-small-nlg-all-crosswoz](https://huggingface.co/ConvLab/mt5-small-nlg-all-crosswoz) | NLG           | CrossWOZ all utterances                |

## Interface

To use trained models in a dialog system, import them through:

```python
from convlab.base_models.t5.nlu import T5NLU
from convlab.base_models.t5.dst import T5DST
from convlab.base_models.t5.nlg import T5NLG

# example instantiation
# model_name_or_path could be model in hugging face hub or local path
nlu = T5NLU(speaker='user', context_window_size=0, model_name_or_path='ConvLab/t5-small-nlu-multiwoz21')
```

See `nlu/nlu.py`, `dst/dst.py`, `nlg/nlg.py` for example usage.

## Support a New Task

To support a new task, you can first serialize model input and output like `create_data.py`, and then train the model with `run_seq2seq.py`. Finally, write a evaluation script for the task or pass the `metric_name_or_path` for an existing metric to `run_seq2seq.py`.

## Author

Qi Zhu(zhuq96 at gmail dot com)