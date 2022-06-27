set -e
n_gpus=1
task_name="lm"
dataset_name=$1
model_type="gpt"
data_dir="data/${task_name}/${dataset_name}/${model_type}"
output_dir="output/${task_name}/${dataset_name}/${model_type}"
cache_dir="../cache"
validation_file="${data_dir}/validation.json"
source_column="dialogue"
max_length=512
model_name_or_path="gpt2-large"
per_device_eval_batch_size=16

python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type ${model_type}
for data_split in validation test train
do
    validation_file="${data_dir}/${data_split}.json"
    dump_eval_loss_to="${data_dir}/${model_name_or_path}_${dataset_name}_${data_split}_token_loss.json"
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
done
