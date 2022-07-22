set -e
n_gpus=8
master_port=23456
task_name="key2gen+key2gen_noisy"
dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+sgd+reddit+wikidialog"
model_type="gpt"
model_name="t5-small"
data_dir="data/${task_name}/${model_type}/${dataset_name}"
output_dir="output/${task_name}/${model_name}/${dataset_name}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
source_column="source"
target_column="target"
truncation_side="left"
max_source_length=512
max_target_length=128
model_name_or_path="${model_name}"
per_device_train_batch_size=64
per_device_eval_batch_size=128
gradient_accumulation_steps=1
num_workers=16
lr=1e-3
num_train_epochs=1

python -m torch.distributed.launch --master_port ${master_port} \
    --nproc_per_node ${n_gpus} ../../t5/run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --save_steps 5000 \
    --save_total_limit 3 \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --preprocessing_num_workers ${num_workers} \
    --dataloader_num_workers ${num_workers} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --optim adafactor \
    --lr_scheduler_type constant \
    --gradient_checkpointing
