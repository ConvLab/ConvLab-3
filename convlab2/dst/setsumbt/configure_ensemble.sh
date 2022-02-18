#!/bin/bash

ENSEMBLE_SIZE=10
DATA_SIZE=7500
SEED=$1
OUT=$2

python run.py --run_nbt \
    --output_dir $OUT \
    --use_descriptions --set_similarity \
    --ensemble_size $ENSEMBLE_SIZE \
    --data_sampling_size $DATA_SIZE \
    --seed $SEED

ENSEMBLE_SIZE=$(($ENSEMBLE_SIZE-1))
for e in $(seq 0 $ENSEMBLE_SIZE);do
    mkdir -p "$OUT/ensemble-$e/dataloaders"

    mv "$OUT/ensemble-$e/train.dataloader" "$OUT/ensemble-$e/dataloaders/"
    cp "$OUT/dataloaders/dev.dataloader" "$OUT/ensemble-$e/dataloaders/"
    cp "$OUT/dataloaders/test.dataloader" "$OUT/ensemble-$e/dataloaders/"
    cp -r $OUT/database "$OUT/ensemble-$e/"
done
