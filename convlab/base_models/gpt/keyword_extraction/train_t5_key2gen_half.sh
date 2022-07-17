set -e
n_gpus=2
master_port=$1
task_name="key2gen"
dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+wikidialog"
model_type="gpt"
split_id=$2
n_splits=$3
data_dir="data/${task_name}/${model_type}/${dataset_name}"
output_dir="output/${task_name}/${model_type}/${dataset_name}_split_${split_id}-of-${n_splits}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train_split_${split_id}-of-${n_splits}.json"
validation_file="${data_dir}/validation_split_${split_id}-of-${n_splits}.json"
test_file="${data_dir}/test_split_${split_id}-of-${n_splits}.json"
source_column="source"
target_column="target"
truncation_side="left"
max_source_length=512
max_target_length=128
model_name_or_path="t5-small"
per_device_train_batch_size=128
per_device_eval_batch_size=128
gradient_accumulation_steps=4
num_workers=16
lr=1e-3
num_train_epochs=1

python -m torch.distributed.launch --master_port ${master_port} \
    --nproc_per_node ${n_gpus} ../../t5/run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers ${num_workers} \
    --dataloader_num_workers ${num_workers} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --adafactor \
    --gradient_checkpointing
