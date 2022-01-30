n_gpus=3
task_name="dialogBIO"
dataset_name="sgd"
data_dir="data/${task_name}/${dataset_name}"
output_dir="output/${task_name}/${dataset_name}"
cache_dir="cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_column="tokens"
target_column="labels"
model_name_or_path="bert-base-uncased"
per_device_train_batch_size=8
per_device_eval_batch_size=16
gradient_accumulation_steps=2
lr=2e-5
num_train_epochs=1
metric_for_best_model="f1"

python create_data.py --tasks ${task_name} --datasets ${dataset_name} --save_dir "data"

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} run_token_classification.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model ${metric_for_best_model} \
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
    --debug underflow_overflow
