set -e
n_gpus=2
task_name="dialogBIO"
dataset_name="multiwoz21"
data_dir="data/${task_name}/${dataset_name}"
output_dir="output/${task_name}/${dataset_name}"
cache_dir="cache"
logging_dir="${output_dir}/runs"
source_column="tokens"
target_column="labels"
model_name_or_path="output/dialogBIO/sgd+tm1+tm2+tm3"
per_device_eval_batch_size=32

python create_data.py --tasks ${task_name} --datasets ${dataset_name} --save_dir "data"

for split in test validation train
do
    CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
        --nproc_per_node ${n_gpus} run_token_classification.py \
        --task_name ${task_name} \
        --train_file ${data_dir}/${split}.json \
        --validation_file ${data_dir}/${split}.json \
        --test_file ${data_dir}/${split}.json \
        --source_column ${source_column} \
        --target_column ${target_column} \
        --model_name_or_path ${model_name_or_path} \
        --do_predict \
        --cache_dir ${cache_dir} \
        --output_dir ${output_dir} \
        --logging_dir ${logging_dir} \
        --overwrite_output_dir \
        --preprocessing_num_workers 4 \
        --per_device_eval_batch_size ${per_device_eval_batch_size}

    mv ${output_dir}/predictions.json ${output_dir}/${split}.json
done

