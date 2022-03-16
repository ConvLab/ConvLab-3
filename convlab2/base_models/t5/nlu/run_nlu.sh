n_gpus=2
task_name="nlu"
dataset_name="multiwoz21"
speaker="user"
context_window_size=0
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
per_device_eval_batch_size=256
gradient_accumulation_steps=1
lr=1e-3
num_train_epochs=10

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --speaker ${speaker} --context_window_size ${context_window_size}

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
    --metric_name_or_path ${metric_name_or_path} \
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
    --debug underflow_overflow \
    --adafactor \
    --gradient_checkpointing
