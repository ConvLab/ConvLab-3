#!/bin/bash

ENSEMBLE_SIZE=10
SEED=$1
OUT=$2

ENSEMBLE_SIZE=$(($ENSEMBLE_SIZE-1))
for e in $(seq 0 $ENSEMBLE_SIZE);do
    python run.py --run_nbt \
        --output_dir "$OUT/ensemble-$e" \
        --use_descriptions --set_similarity \
        --do_train --do_eval \
        --seed $SEED
done
