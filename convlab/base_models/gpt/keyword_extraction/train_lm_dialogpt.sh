set -e
n_gpus=1
task_name="lm"
dataset_name="multiwoz21"
model_type="dialogpt"
data_dir="data/${task_name}/${dataset_name}/${model_type}"
output_dir="output/${task_name}/${dataset_name}/${model_type}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_column="dialogue"
max_length=512
model_name_or_path="microsoft/DialoGPT-large"
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=4
lr=5e-5
num_train_epochs=3

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type ${model_type}

python ../run_clm.py \
    --model_name_or_path ${model_name_or_path} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --max_length ${max_length} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --gradient_checkpointing
