set -e
n_gpus=2
master_port=23457
task_name="key2gen_noisy"
dataset_name=$1
model_type="gpt"
data_dir="data/${task_name}/${model_type}/${dataset_name}"
output_dir="output/${task_name}/${model_type}/${dataset_name}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_column="source"
target_column="target"
truncation_side="left"
max_source_length=512
max_target_length=128
model_name_or_path="output/${task_name}/${model_type}/dailydialog+metalwoz+tm1+tm2+tm3"
per_device_train_batch_size=128
per_device_eval_batch_size=128
gradient_accumulation_steps=2
lr=1e-3
num_train_epochs=3

python -m torch.distributed.launch --master_port ${master_port} \
    --nproc_per_node ${n_gpus} ../../t5/run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_predict \
    --predict_with_generate \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --adafactor \
    --gradient_checkpointing