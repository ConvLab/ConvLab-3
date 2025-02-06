n_gpus=8
task_name="coqr"
dataset_name="canard"
master_port=$1

python create_data.py -t coqr -d canard -w data/canard

for task_type in origin reverse_SDI
do
    data_dir="data/${dataset_name}/${task_type}"
    output_dir="output/${dataset_name}/${task_type}"
    cache_dir="../cache"
    logging_dir="${output_dir}/runs"
    train_file="${data_dir}/train.json"
    validation_file="${data_dir}/validation.json"
    test_file="${data_dir}/test.json"
    metric_name_or_path="../nlg/nlg_metric.py"
    source_column="input"
    target_column="output"
    truncation_side="left"
    max_source_length=1024
    max_target_length=512
    model_name_or_path="${PRETRAINED_MODELS}/t5-large"
    per_device_train_batch_size=64
    per_device_eval_batch_size=64
    gradient_accumulation_steps=1
    lr=1e-3
    num_train_epochs=1

    python -m torch.distributed.launch \
        --nproc_per_node ${n_gpus} \
        --master_port ${master_port} ../run_seq2seq.py \
        --task_name ${task_name} \
        --train_file ${train_file} \
        --validation_file ${validation_file} \
        --test_file ${test_file} \
        --source_column ${source_column} \
        --target_column ${target_column} \
        --max_source_length ${max_source_length} \
        --max_target_length ${max_target_length} \
        --truncation_side ${truncation_side} \
        --model_name_or_path ${model_name_or_path} \
        --do_train \
        --do_eval \
        --do_predict \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 3 \
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
done