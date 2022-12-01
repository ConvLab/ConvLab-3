# Introduction

This is the TripPy DST module for ConvLab-3.

## Supported encoders

* RoBERTa
* BERT (full support w.i.p.)
* ELECTRA (full support w.i.p.)

## Supported datasets

* MultiWOZ 2.X
* Unified Data Format

## Requirements

* transformers (tested: 4.18.0)
* torch (tested: 1.8.0)

# Parameters

```
model_type # Default: "roberta", Type of the model (Supported: "roberta", "bert", "electra")
model_name # Default: "roberta-base", Name of the model (Use -h to print a list of names)
model_path # Path to a model checkpoint. Note, this can also be a HuggingFace model
dataset_name # Default: "multiwoz21", Name of the dataset the model was trained on and/or is being applied to
local_files_only # Default: False, Set to True to load local files only. Useful for offline systems 
nlu_usr_config # Path to a NLU config file. Only needed for internal evaluation
nlu_sys_config # Path to a NLU config file. Only needed when using word-level policies
nlu_usr_path # Path to a NLU model file. Only needed for internal evaluation
nlu_sys_path # Path to a NLU model file. Only needed when using word-level policies
no_eval # Default: True, Set to False if internal evaluation should be conducted
no_history # Default: False, Set to True if dialogue history should be omitted during inference
```

# Model checkpoint

A model checkpoint can either be trained from scratch using the TripPy codebase (see below), or a ready-to-use checkpoint can be loaded from the [HuggingFace repository](https://huggingface.co/ConvLab) for ConvLab.

Currently, the following checkpoint is available to be loaded from HuggingFace:

```
ConvLab/roberta-base-trippy-dst-multiwoz21
```

To load this checkpoint, use the following parameters for TripPy DST in ConvLab-3:

```
model_type="roberta"
model_name="roberta-base"
model_path="ConvLab/roberta-base-trippy-dst-multiwoz21"
```

The checkpoint will be downloaded and cached automatically.

# Training

TripPy can easily be trained for the abovementioned supported datasets using the original code in the official [TripPy repository](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public). Simply clone the code and run the appropriate DO.* script to train a TripPy DST. After training, set model_path to the preferred checkpoint to use TripPy in ConvLab-3.

# Training and evaluation with PPO policy

Switch to the directory:
```
cd ../../policy/ppo
```

Edit trippy_config.json accordingly, e.g., edit paths to model checkpoints.

For training, run
```
train.py --path trippy_config.json
```

For evaluation, set training epochs to 0.

# Paper

[TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://aclanthology.org/2020.sigdial-1.4/)
