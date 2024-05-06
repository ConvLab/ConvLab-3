n_gpus=4
task_name="coqr"
dataset_name=$1
src_data_dir=$2
read_data_dir=$3
output_dir=$4
model_type=$5
context_window_size=$6
master_port=$7
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${read_data_dir}/coqr/train.json"
validation_file="${read_data_dir}/coqr/validation.json"
infer_test_single_file="${read_data_dir}/coqr/test_single_domain.json"
infer_test_multi_file="${read_data_dir}/coqr/test_multi_domain.json"
metric_name_or_path="../nlg/nlg_metric.py"
source_column="input"
target_column="output"
truncation_side="left"
max_source_length=1024
max_target_length=512
model_name_or_path="output/canard/origin"
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=2
lr=1e-3
num_train_epochs=3

# Train CoQR model on SYN data
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
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --predict_with_generate \
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

for test_file in ${infer_test_single_file} ${infer_test_multi_file}
do
    python -m torch.distributed.launch \
        --nproc_per_node ${n_gpus} \
        --master_port ${master_port} ../run_seq2seq.py \
        --task_name ${task_name} \
        --test_file ${test_file} \
        --source_column ${source_column} \
        --target_column ${target_column} \
        --max_source_length ${max_source_length} \
        --max_target_length ${max_target_length} \
        --truncation_side ${truncation_side} \
        --model_name_or_path ${output_dir} \
        --do_predict \
        --predict_with_generate \
        --generation_num_beams 5 \
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
done

python evaluate_coqr_infer.py -p ${output_dir}/test_single_domain_generated_predictions.json -i ${read_data_dir}/test_single_domain.json -d ${read_data_dir}
python evaluate_coqr_infer.py -p ${output_dir}/test_multi_domain_generated_predictions.json -i ${read_data_dir}/test_multi_domain.json -d ${read_data_dir}


# CoQR model trained on canard infer on test set
output_dir=${model_name_or_path}

for test_file in ${infer_test_single_file} ${infer_test_multi_file}
do
    python -m torch.distributed.launch \
        --nproc_per_node ${n_gpus} \
        --master_port ${master_port} ../run_seq2seq.py \
        --task_name ${task_name} \
        --test_file ${test_file} \
        --source_column ${source_column} \
        --target_column ${target_column} \
        --max_source_length ${max_source_length} \
        --max_target_length ${max_target_length} \
        --truncation_side ${truncation_side} \
        --model_name_or_path ${output_dir} \
        --do_predict \
        --predict_with_generate \
        --generation_num_beams 5 \
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
done

python evaluate_coqr_infer.py -p ${output_dir}/test_single_domain_generated_predictions.json -i ${read_data_dir}/test_single_domain.json -d ${src_data_dir}
python evaluate_coqr_infer.py -p ${output_dir}/test_multi_domain_generated_predictions.json -i ${read_data_dir}/test_multi_domain.json -d ${src_data_dir}


# chatgpt
python evaluate_coqr_infer.py -p ${src_data_dir}/test_multi_domain_chatgpt.json -i ${read_data_dir}/test_multi_domain.json -d ${src_data_dir}

# DST inference
if [ "${dataset_name}" == "sgd" ]
then
    echo sgd
    output_prefix="output0615/${dataset_name}/group0/qadst_f1_th>0.1"
else
    echo multiwoz
    output_prefix="output0615/${dataset_name}/qadst_f1_th>0.1"
fi
output_dir="${output_prefix}/t5-large_aug2_x2.0_model${model_type}_context${context_window_size}"
task_name="dst"

test_files=(${src_data_dir}/test_multi_domain_coqr_canard.json ${src_data_dir}/test_multi_domain_coqr_chatgpt.json ${read_data_dir}/test_multi_domain_coqr.json)
output_test_files=(${output_dir}/test_multi_domain_coqr_canard_generated_predictions.json ${output_dir}/test_multi_domain_coqr_chatgpt_generated_predictions.json ${output_dir}/test_multi_domain_coqr_generated_predictions.json)

for i in 0 1 2
do
    test_file=${test_files[i]}
    python -m torch.distributed.launch \
        --nproc_per_node ${n_gpus} \
        --master_port ${master_port} inference.py \
        --context_window_size ${context_window_size} \
        --task_name ${task_name} \
        --dataset_name ${dataset_name} \
        --test_file ${test_file} \
        --src_data_dir ${src_data_dir} \
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

    output_test_file=${output_test_files[i]}

    python evaluate.py -d ${dataset_name} -p ${output_test_file} -i ${src_data_dir}/test_multi_domain.json
done