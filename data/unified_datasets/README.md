# Unified data format

## Overview
We transform different datasets into a unified format under `data/unified_datasets` directory. One could also access processed datasets from Hugging Face's `Dataset`:
```python
from datasets import load_dataset
dataset = load_dataset('ConvLab/$dataset')
```

Each dataset contains at least these files:

- `README.md`: dataset description and the **main changes** from original data to processed data. Should include the instruction on how to get the original data and transform them into the unified format.
- `preprocess.py`: python script that transform the original data into the unified format. By running `python preprocess.py` we can get `data.zip` that contains all data. The structure `preprocess.py` should be like:

```python
def preprocess():
    pass
if __name__ == '__main__':
    preprocess()
```

- `data.zip`: (also available in https://huggingface.co/ConvLab) the zipped directory contains:
  - `ontology.json`: dataset ontology, contains descriptions, state definition, etc.
  - `dialogues.json`, a list of all dialogues in the dataset.
  - other necessary files such as databases.

Datasets that require database interaction should also include the following file:
- `database.py`: load the database and define the query function:
```python
class Database:
    def __init__(self):
        """extract data.zip and load the database."""

    def query(self, domain:str, state:dict, topk:int, **kwargs)->list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""
```

## Unified format
We first introduce the unified format of `ontology` and `dialogues`. To transform a new dataset into the unified format:
1. Create `data/unified_datasets/$dataset` folder, where `$dataset` is the name of the dataset.
2. Write `preprocess.py` to transform the original dataset into the unified format, producing `data.zip`.
3. Run `python test.py $dataset` in the `data/unified_datasets` directory to check the validation of processed dataset and get data statistics.
4. Write `README.md` to describe the data.

### Ontology

`ontology.json`: a *dict* containing:

- `domains`: (*dict*) descriptions for domains, slots. Must contains all slots in the state and non-binary dialogue acts.
  - `$domain_name`: (*dict*)
    - `description`: (*str*) description for this domain.
    - `slots`: (*dict*)
      - `$slot_name`: (*dict*)
        - `description`: (*str*) description for this slot.
        - `is_categorical`: (*bool*) categorical slot or not.
        - `possible_values`: (*list*) List of possible values the slot can take. If the slot is a categorical slot, it is a complete list of all the possible values. If the slot is a non categorical slot, it is either an empty list or a small sample of all the values taken by the slot.

- `intents`: (*dict*) descriptions for intents.
  - `$intent_name`: (*dict*)
    - `description`: (*str*) description for this intent.
- `binary_dialogue_acts`: (*list* of *dict*) binary dialogue act is a more detailed intent where the value is not extracted from dialogues, e.g. request the address of a hotel.
  - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. domain, slot, and value may be empty.
- `state`: (*dict*) dialogue state of all domains.
  - `$domain_name`: (*dict*)
    - `$slot_name: ""`: slot with empty value. Note that the slot set are the subset of the slot set in Part 1 definition.

### Dialogues

`data.json`: a *list* of dialogues (*dict*) containing:

- `dataset`: (*str*) dataset name, must be the same as the data directory.
- `data_split`: (*str*) in `["train", "validation", "test"]`.
- `dialogue_id`: (*str*) `"$dataset-$split-$id"`, `id` increases from 0.
- `domains`: (*list*) involved domains in this dialogue.
- `goal`: (*dict*, optional)
  - `description`: (*str*) a string describes the user goal.
  - `constraints`: (*dict*, optional) same format as dialogue state of involved domains but with only filled slots as constraints.
  - `requirements`: (*dict*, optional) same format as dialogue state of involved domains but with only empty required slots.
- `turns`: (*list* of *dict*)
  - `speaker`: (*str*) "user" or "system".
  - `utterance`: (*str*)
  - `utt_idx`: (*int*) `turns['utt_idx']` gives current turn.
  - `dialogue_acts`: (*dict*, optional)
    - `categorical`: (*list* of *dict*) for categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology.
    - `non-categorical` (*list* of *dict*) for non-categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str), "start": (int), "end": (int)}`. `start` and `end` are character indexes for the value span in the utterance and can be absent.
    - `binary` (*list* of *dict*) for binary dialogue acts in ontology.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Possible dialogue acts are listed in the `ontology['binary_dialogue_acts']`.
  - `state`: (*dict*, user side, optional) dialogue state of involved domains. full state is shown in `ontology['state']`.
    - `$domain_name`: (*dict*) contains all slots in this domain.
      - `$slot_name`: (*str*) value for this slot.
  - `db_results`: (*dict*, optional)
    - `$domain_name`: (*list* of *dict*) topk entities (each entity contains slot-value pairs)

Other attributes are optional.

Run `python test.py $dataset` in the `data/unified_datasets` directory to check the validation of processed dataset and get data statistics.

### README
Each dataset has a README.md to describe the original and transformed data. Follow the Hugging Face's [dataset card creation](https://huggingface.co/docs/datasets/dataset_card.html) to export `README.md`. Make sure that the following additional information is included in the **Dataset Summary** section:
- Main changes from original data to processed data.
- Annotations: whether have user goal, dialogue acts, state, db results, etc.

And the data statistics given by `test.py` should be included in the **Data Splits** section.

## Example dialogue of Schema-Guided Dataset
