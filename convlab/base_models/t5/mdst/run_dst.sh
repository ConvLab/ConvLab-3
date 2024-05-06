n_gpus=4
task_name="dst"
dataset_name=$1
model_size=$2
model_type=$3
src_data_dir=$4
read_data_dir=$5
output_dir=$6
context_window_size=$7
master_port=$8
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${read_data_dir}/model${model_type}_context${context_window_size}/train.json"
validation_file="${read_data_dir}/model${model_type}_context${context_window_size}/validation.json"
source_column="input"
target_column="output"
truncation_side="left"
max_source_length=1024
max_target_length=512
model_name_or_path="${PRETRAINED_MODELS}/${model_size}"
per_device_train_batch_size=32
per_device_eval_batch_size=32
gradient_accumulation_steps=1
lr=1e-3
num_train_epochs=5

if ((${model_type}>3))
then
    test_file_single="${read_data_dir}/test_single_domain.json"
    test_file_multi="${read_data_dir}/test_multi_domain.json"
else
    test_file_single="${src_data_dir}/test_single_domain.json"
    test_file_multi="${src_data_dir}/test_multi_domain.json"
fi

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
    --master_port ${master_port} inference.py \
    --context_window_size ${context_window_size} \
    --task_name ${task_name} \
    --dataset_name ${dataset_name} \
    --src_data_dir ${src_data_dir} \
    --test_file ${test_file_single} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${output_dir} \
    --model_type ${model_type} \
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
    --master_port ${master_port} inference.py \
    --context_window_size ${context_window_size} \
    --task_name ${task_name} \
    --dataset_name ${dataset_name} \
    --src_data_dir ${src_data_dir} \
    --test_file ${test_file_multi} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${output_dir} \
    --model_type ${model_type} \
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

python evaluate.py -d ${dataset_name} -p ${output_dir}/test_single_domain_generated_predictions.json
python evaluate.py -d ${dataset_name} -p ${output_dir}/test_multi_domain_generated_predictions.json
