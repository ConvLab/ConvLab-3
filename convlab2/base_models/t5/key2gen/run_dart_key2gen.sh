n_gpus=1
task_name="dart"
dataset_name="dart"
speaker="system"
context_window_size=0
data_dir="data/${task_name}/key2gen_${dataset_name}"
output_dir="output/${task_name}/key2gen_${dataset_name}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
metric_name_or_path="../nlg/nlg_metric.py"
metric_for_best_model="bleu"
source_column="triples"
target_column="text"
truncation_side="left"
max_source_length=512
max_target_length=512
model_name_or_path="t5-small"
per_device_train_batch_size=128
per_device_eval_batch_size=64
gradient_accumulation_steps=4
lr=1e-3
num_train_epochs=10

python create_data_key2gen.py -t ${task_name} -d ${dataset_name} -s ${speaker} -c ${context_window_size} --key2gen

python ../run_seq2seq.py \
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
    --save_total_limit 3 \
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
    --adafactor \
    --gradient_checkpointing

python ../run_seq2seq.py \
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
    --adafactor \
    --gradient_checkpointing

python ../nlg/merge_predict_res.py -d ${dataset_name} -s ${speaker} -c ${context_window_size} -p ${output_dir}/generated_predictions.json

python ../../../nlg/evaluate_unified_datasets.py -p ${output_dir}/predictions.json --dataset_name ${dataset_name}
