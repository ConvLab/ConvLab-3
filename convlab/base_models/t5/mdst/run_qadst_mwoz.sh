n_gpus=8
task_name="qadst"
dataset_name="multiwoz21"
master_port=$1
data_dir="data/${dataset_name}/single_domain_qadst"
output_dir="output/${dataset_name}/single_domain_qadst"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
infer_train_single_domain_file="${data_dir}/train_single_domain_qa.json"
infer_validation_single_domain_file="${data_dir}/validation_single_domain_qa.json"
source_column="input"
target_column="output"
truncation_side="left"
max_source_length=1024
max_target_length=32
model_name_or_path="${PRETRAINED_MODELS}/unifiedqa-v2-t5-large-1251000"
per_device_train_batch_size=32
per_device_eval_batch_size=64
gradient_accumulation_steps=4
lr=1e-3
num_train_epochs=3

python create_data.py -t single_domain_qadst -d ${dataset_name} -w ${data_dir}

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} \
    --master_port ${master_port} ../run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
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

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} \
    --master_port ${master_port} ../run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${infer_train_single_domain_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${output_dir} \
    --do_predict \
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
    --optim adafactor \
    --gradient_checkpointing

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} \
    --master_port ${master_port} ../run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${infer_validation_single_domain_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${output_dir} \
    --do_predict \
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
    --optim adafactor \
    --gradient_checkpointing

python evaluate_qa.py -p ${output_dir} -d "data/${dataset_name}"
