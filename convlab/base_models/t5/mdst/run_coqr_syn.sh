n_gpus=4
task_name="coqr"
dataset_name=$1
master_port=$2
data_dir=$3
output_dir=${data_dir}
cache_dir="../cache"
logging_dir="${output_dir}/runs"
infer_train_coqr_file="${data_dir}/train_aug_dials4coqr.json"
infer_validation_coqr_file="${data_dir}/validation_aug_dials4coqr.json"
metric_name_or_path="../nlg/nlg_metric.py"
source_column="input"
target_column="output"
truncation_side="left"
max_source_length=1024
max_target_length=512
model_name_or_path="output/canard/reverse_SDI"
per_device_train_batch_size=64
per_device_eval_batch_size=32
gradient_accumulation_steps=1
lr=1e-3
num_train_epochs=1

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} \
    --master_port ${master_port} ../run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${infer_train_coqr_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_predict \
    --predict_with_generate \
    --do_sample \
    --metric_name_or_path ${metric_name_or_path} \
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
    --test_file ${infer_validation_coqr_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_predict \
    --predict_with_generate \
    --do_sample \
    --metric_name_or_path ${metric_name_or_path} \
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

python evaluate_qr.py -p ${data_dir}/validation_aug_dials4coqr_generated_predictions.json -o ${data_dir}/validation_aug_dials.json
python evaluate_qr.py -p ${data_dir}/train_aug_dials4coqr_generated_predictions.json -o ${data_dir}/train_aug_dials.json
