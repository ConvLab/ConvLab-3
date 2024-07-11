n_gpus=4
task_name="dom_cls"
dataset_name=$1
model_size=$2
src_data_dir=$3
read_data_dir=$4
output_dir=$5
context_window_size=$6
master_port=$7
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${read_data_dir}/dom_cls_c${context_window_size}/train.json"
validation_file="${read_data_dir}/dom_cls_c${context_window_size}/validation.json"
test_file_single="${read_data_dir}/dom_cls_c${context_window_size}/test_single_domain.json"
test_file_multi="${read_data_dir}/dom_cls_c${context_window_size}/test_multi_domain.json"
source_column="input"
target_column="output"
truncation_side="left"
max_source_length=1024
max_target_length=512
model_name_or_path="${PRETRAINED_MODELS}/t5-large"
per_device_train_batch_size=32
per_device_eval_batch_size=32
gradient_accumulation_steps=1
lr=1e-3
num_train_epochs=3

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
    --test_file ${test_file_single} \
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
    --test_file ${test_file_multi} \
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

python evaluate_dom_cls.py -p ${output_dir}/test_single_domain_generated_predictions.json -i ${src_data_dir}/test_single_domain.json -d ${read_data_dir}
python evaluate_dom_cls.py -p ${output_dir}/test_multi_domain_generated_predictions.json -i ${src_data_dir}/test_multi_domain.json -d ${read_data_dir}
