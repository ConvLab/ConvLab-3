n_gpus=1
task_name="nlu"
dataset_name="tm3"
speaker="user"
context_window_size=3
data_dir="data/${task_name}/${dataset_name}/${speaker}/context_${context_window_size}"
output_dir="output/${task_name}/${dataset_name}/${speaker}/context_${context_window_size}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
metric_name_or_path="nlu_metric.py"
metric_for_best_model="overall_f1"
source_prefix="${data_dir}/source_prefix.txt"
source_column="context"
target_column="dialogue_acts_seq"
model_name_or_path="t5-small"
per_device_train_batch_size=128
per_device_eval_batch_size=64
gradient_accumulation_steps=2
lr=1e-3
num_train_epochs=10

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --speaker ${speaker} --context_window_size ${context_window_size}

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} ../run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --source_prefix ${source_prefix} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --save_strategy epoch \
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
    --debug underflow_overflow \
    --adafactor \
    --gradient_checkpointing

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} ../run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --source_prefix ${source_prefix} \
    --model_name_or_path ${output_dir} \
    --do_predict \
    --predict_with_generate \
    --metric_name_or_path ${metric_name_or_path} \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
