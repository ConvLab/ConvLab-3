#!/bin/bash

# This step is needed when using a different configuration.
# It can be commented out after removing data from a previous configuration and training
# using various seeds.
rm -rdf ../../../data_loaders/processed_utterance_data

# This step can be commented out once a supervised starting point has been obtained
# for a specific configuration
python train_supervised.py \
    --utterance_level \
    --setsumbt_path "https://cloud.cs.uni-duesseldorf.de/s/Yqkzz8NW3yoMWRk/download/setsumbt_end.zip"

# This is the main RL step
python train.py \
    --path "setsumbt_end_baseline_config.json" \
    --seed 20211202
