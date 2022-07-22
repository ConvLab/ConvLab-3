n_gpus=1
task_name="goal2dialogue"
dataset_name="multiwoz21"
data_dir="data/${task_name}/${dataset_name}"
output_dir="output/${task_name}/${dataset_name}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_column="goal"
target_column="dialogue"
max_target_length=1024
model_name_or_path="t5-small"
per_device_train_batch_size=32
per_device_eval_batch_size=64
gradient_accumulation_steps=4
lr=1e-3
num_train_epochs=10

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name}

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} ../run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_target_length ${max_target_length} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --do_predict \
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
    --optim adafactor \
    --gradient_checkpointing
