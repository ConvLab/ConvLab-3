set -e
n_gpus=1
task_name="lm"
dataset_name="multiwoz21"
model_type="dialogpt"
data_dir="data/${task_name}/${dataset_name}/${model_type}"
output_dir="output/${task_name}/${dataset_name}/${model_type}"
cache_dir="../cache"
validation_file="${data_dir}/validation.json"
source_column="dialogue"
max_length=512
model_name_or_path="microsoft/DialoGPT-large"
per_device_eval_batch_size=4

dump_eval_loss_to="${data_dir}/dialogpt-large_${dataset_name}_token_loss.json"
python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type dialogpt
python ../run_clm.py \
    --dump_eval_loss_to ${dump_eval_loss_to}\
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${data_dir} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --max_length ${max_length} \
    --do_eval \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size ${per_device_eval_batch_size}
python lmloss2keywords.py --token_loss_file ${dump_eval_loss_to} --model_type ${model_type}

dump_eval_loss_to="${data_dir}/dialogpt-large-mwoz_${dataset_name}_token_loss.json"
python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type dialogpt
python ../run_clm.py \
    --dump_eval_loss_to ${dump_eval_loss_to}\
    --model_name_or_path ${output_dir} \
    --output_dir ${data_dir} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --max_length ${max_length} \
    --do_eval \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size ${per_device_eval_batch_size}
python lmloss2keywords.py --token_loss_file ${dump_eval_loss_to} --model_type ${model_type}

model_type="gpt"
data_dir="data/${task_name}/${dataset_name}/${model_type}"
validation_file="${data_dir}/validation.json"
model_name_or_path="gpt2-large"
dump_eval_loss_to="${data_dir}/gpt2-large_${dataset_name}_token_loss.json"
python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type gpt
python ../run_clm.py \
    --dump_eval_loss_to ${dump_eval_loss_to}\
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${data_dir} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --max_length ${max_length} \
    --do_eval \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size ${per_device_eval_batch_size}
python lmloss2keywords.py --token_loss_file ${dump_eval_loss_to} --model_type ${model_type}
