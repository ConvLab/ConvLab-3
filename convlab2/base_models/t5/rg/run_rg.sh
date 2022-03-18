n_gpus=8
task_name="rg"
dataset_name="multiwoz21"
speaker="system"
data_dir="data/${task_name}/${dataset_name}/${speaker}"
output_dir="output/${task_name}/${dataset_name}/${speaker}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_prefix="${data_dir}/source_prefix.txt"
source_column="context"
target_column="response"
model_name_or_path="t5-small"
per_device_train_batch_size=32
per_device_eval_batch_size=128
gradient_accumulation_steps=1
lr=1e-3
num_train_epochs=5

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --speaker ${speaker}

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} ../run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --source_prefix ${source_prefix} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --predict_with_generate \
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
    --debug underflow_overflow \
    --adafactor \
    --gradient_checkpointing
