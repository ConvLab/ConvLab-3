# Unified data format

## Usage
We transform different datasets into a unified format under `data/unified_datasets` directory. To import a unified datasets:

```python
from convlab.util import load_dataset, load_ontology, load_database

dataset = load_dataset('multiwoz21')
ontology = load_ontology('multiwoz21')
database = load_database('multiwoz21')
```

`dataset` is a dict where the keys are data splits and the values are lists of dialogues. `database` is an instance of `Database` class that has a `query` function. The format of dialogue, ontology, and Database are defined below.

We provide a function `load_unified_data` to transform the dialogues into turns as samples. By passing different arguments to `load_unified_data`, we provide functions to load data for different components:

```python
from convlab.util import load_unified_data, load_nlu_data, load_dst_data, load_policy_data, load_nlg_data, load_e2e_data

nlu_data = load_nlu_data(dataset, data_split='test', speaker='user')
dst_data = load_dst_data(dataset, data_split='test', speaker='user', context_window_size=5)
```

To customize the data loading process, see the definition of `load_unified_data`.

## Unified datasets
Each dataset contains at least these files:

- `README.md`: dataset description and the **main changes** from original data to processed data. Should include the instruction on how to get the original data and transform them into the unified format.
- `preprocess.py`: python script that transform the original data into the unified format. By running `python preprocess.py` we can get `data.zip` and `dummy_data.json`. The structure `preprocess.py` should be like:

```python
def preprocess():
    pass
if __name__ == '__main__':
    preprocess()
```

- `data.zip`: the zipped directory `data` contains:
  - `ontology.json`: dataset ontology, contains descriptions, state definition, etc.
  - `dialogues.json`: a list of all dialogues in the dataset.
  - other necessary files such as databases.
- `dummy_data.json`: a list of 10 dialogues from `dialogues.json` for illustration.
- `shuffled_dial_ids.json`: 10 random shuffled data orders created by `check.py` for experiment reproducibility, can be used in `load_dataset` function by passing the `dial_ids_order` in [0, 9]

Datasets that require database interaction should also include the following file:
- `database.py`: load the database and define the query function:
```python
from convlab.util.unified_datasets_util import BaseDatabase

class Database(BaseDatabase):
    def __init__(self):
        """extract data.zip and load the database."""

    def query(self, domain:str, state:dict, topk:int, **kwargs)->list:
        """return a list of topk entities (dict containing slot-value pairs) for a given domain based on the dialogue state."""
```

## Unified format
We first introduce the unified format of `ontology` and `dialogues`. To transform a new dataset into the unified format:
1. Create `data/unified_datasets/$dataset` folder, where `$dataset` is the name of the dataset.
2. Write `preprocess.py` to transform the original dataset into the unified format, producing `data.zip` and `dummy_data.json`.
3. Run `python check.py $dataset` in the `data/unified_datasets` directory to check the validation of processed dataset and get data statistics and shuffled dialog ids.
4. Write `README.md` to describe the data following [How to create dataset README](#how-to-create-dataset-readme).

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
- `dialogue_acts`: (*dict*) dialogue act dictionaries extracted from the data, separated by their types. Each dialogue act is a *str* converted by a *dict* like `"{'user': True, 'system': True, 'intent': 'inform', 'domain': 'attraction', 'slot': 'area'}"` that includes intent, domain, slot, and whether the speakers use this dialogue act.
  - `categorical`: (*list* of *str*) dictionary for categorical dialogue acts.
  - `non-categorical`: (*list* of *str*) dictionary for non-categorical dialogue acts.
  - `binary`: (*list* of *str*) dictionary for binary dialogue acts that are more detailed intents without values, e.g. request the address of a hotel.  Note that the `slot` in a binary dialogue act may not be an actual slot that presents in `ontology['domains'][domain]['slots']`.
- `state`: (*dict*) dialogue state of all domains.
  - `$domain_name`: (*dict*)
    - `$slot_name: ""`: slot with empty value. Note that the slot set are the subset of the slot set in Part 1 definition.

### Dialogues

`dialogues.json`: a *list* of dialogues (*dict*) containing:

- `dataset`: (*str*) dataset name, must be the same as the data directory.
- `data_split`: (*str*) in `["train", "validation", "test", ...]`.
- `dialogue_id`: (*str*) `"$dataset-$split-$id"`, `id` increases from 0.
- `domains`: (*list*) involved domains in this dialogue.
- `goal`: (*dict*)
  - `description`: (*str*, could be empty) a string describes the user goal.
  - `inform`: (*dict*, could be empty) same format as dialogue state of involved domains but with only filled slots as constraints.
  - `request`: (*dict*, could be empty) same format as dialogue state of involved domains but with only empty requested slots.
- `turns`: (*list* of *dict*)
  - `speaker`: (*str*) "user" or "system".
  - `utterance`: (*str*)
  - `utt_idx`: (*int*) `turns['utt_idx']` gives current turn.
  - `dialogue_acts`: (*dict*)
    - `categorical`: (*list* of *dict*, could be empty) for categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str)}`. Value sets are defined in the ontology.
    - `non-categorical` (*list* of *dict*, could be empty) for non-categorical slots.
      - `{"intent": (str), "domain": (str), "slot": (str), "value": (str), "start": (int), "end": (int)}`. `start` and `end` are character indexes for the value span in the utterance and can be absent.
    - `binary` (*list* of *dict*, could be empty) for binary dialogue acts in ontology.
      - `{"intent": (str), "domain": (str), "slot": (str)}`. Binary dialogue acts are more detailed intents without values, e.g. request the address of a hotel.
  - `state`: (*dict*, user side, could be empty) dialogue state of involved domains. full state is shown in `ontology['state']`.
    - `$domain_name`: (*dict*) contains all slots in this domain.
      - `$slot_name`: (*str*) value for this slot.
  - `db_results`: (*dict*, system side, could be empty)
    - `$domain_name`: (*list* of *dict*) topk entities (each entity contains slot-value pairs)

Note that multiple descriptions/values are separated by `"|"`.

Other attributes are optional.

> **Necessary**: Run `python check.py $dataset` in the `data/unified_datasets` directory to check the validation of processed dataset and get data statistics in `data/unified_datasets/$dataset/stat.txt` as well as shuffled dialog ids in `data/unified_datasets/$dataset/shuffled_dial_ids.json`.

### How to create dataset README
Each dataset has a README.md to describe the original and transformed data. Please follow the `README_TEMPLATE.md` and make sure that you:
- include your name and email in **Who transforms the dataset**.
- include the following additional information in the **Dataset Summary** section:
  - How to get the transformed data from original data.
  - Main changes of the transformation.
  - Annotations: whether has user goal, dialogue acts, state, db results, etc.
- include the data statistics given by `check.py` (in `data/unified_datasets/$dataset/stat.txt`) in the **Data Splits** section.
