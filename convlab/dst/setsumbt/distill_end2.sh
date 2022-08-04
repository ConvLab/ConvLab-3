#!/bin/bash

ENSEMBLE_SIZE=10
SEED=$1
OUT=$2

ENSEMBLE_SIZE=$(($ENSEMBLE_SIZE-1))
for e in $(seq 0 $ENSEMBLE_SIZE);do
    cp "$OUT/ensemble-$e/pytorch_model.bin" "$OUT/pytorch_model_$e.bin"
done
cp "$OUT/ensemble-0/config.json" "$OUT/config.json"

for SET in "train" "dev" "test";do
    python distillation_setup.py --get_ensemble_distributions \
        --model_path $OUT \
        --model_type roberta \
        --set_type $SET \
        --ensemble_size $ENSEMBLE_SIZE \
        --reduction none
done

python distillation_setup.py --build_dataloaders \
    --model_path $OUT \
    --set_type train \
    --batch_size 3

for SET in "dev" "test";do
    python distillation_setup.py --build_dataloaders \
        --model_path $OUT \
        --set_type $SET \
        --batch_size 16
done

python run.py --run_nbt \
    --output_dir $OUT \
    --loss_function "distribution_distillation" \
    --use_descriptions --set_similarity \
    --do_train --do_eval \
    --seed $SEED
