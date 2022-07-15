n_gpus=4
master_port=23456
task_name="lm"
model_type="gpt"
cache_dir="../cache"
source_column="dialogue"
max_length=512
model_name_or_path="/data/zhuqi/pre-trained-models/gpt2-large"
per_device_eval_batch_size=16

for dataset_name in dailydialog metalwoz tm1 tm2 tm3 sgd reddit wikidialog
do
    data_dir="data/${task_name}/${model_type}/${dataset_name}"
    output_dir="output/${task_name}/${model_type}/${dataset_name}"

    python ../create_data.py --tasks ${task_name} --datasets ${dataset_name} --model_type ${model_type}
    for data_split in validation train
    do
        validation_file="${data_dir}/${data_split}.json"
        dump_eval_loss_to="${data_dir}/token_loss_${data_split}.json"
        rm ${dump_eval_loss_to}
        python -m torch.distributed.launch --master_port ${master_port} \
            --nproc_per_node ${n_gpus} ../run_clm.py \
            --dump_eval_loss_to ${dump_eval_loss_to}\
            --model_name_or_path ${model_name_or_path} \
            --output_dir ${data_dir} \
            --validation_file ${validation_file} \
            --source_column ${source_column} \
            --max_length ${max_length} \
            --do_eval \
            --cache_dir ${cache_dir} \
            --preprocessing_num_workers 4 \
            --per_device_eval_batch_size ${per_device_eval_batch_size}
    done
done
