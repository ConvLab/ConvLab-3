set -e
dataset_path=$1
model_name=$2
model_name_or_path=$3
dataset_name=$4
if [ "${dataset_name}" == "multiwoz21" ]
then
    task_name="nlg"
else
    task_name=${dataset_name}
fi
master_port=$5

n_gpus=2
cache_dir="../cache"
metric_name_or_path="metric.py"
source_column="context+knowledge"
target_column="response"
truncation_side="left"
max_source_length=512
max_target_length=512
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=1
num_workers=16
lr=1e-3
num_train_epochs=100

for shot in 50 100 200
do
    for dial_ids_order in 0 1 2 3 4
    do
        python create_data.py -t ${task_name} -d ${dataset_name} -o ${dial_ids_order} -s ${shot}

        data_dir="data/${task_name}/${dataset_name}_${shot}shot_order${dial_ids_order}"
        output_dir="output/${model_name}/${task_name}/${dataset_name}_${shot}shot_order${dial_ids_order}"
        logging_dir="${output_dir}/runs"
        train_file="${data_dir}/train.json"
        validation_file="${data_dir}/validation.json"

        # training
        python -m torch.distributed.launch --master_port ${master_port} \
            --nproc_per_node ${n_gpus} ../run_seq2seq.py \
            --task_name ${task_name} \
            --dataset_name ${dataset_path} \
            --dataset_config_name ${task_name} \
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
            --load_best_model_at_end \
            --overwrite_output_dir \
            --cache_dir ${cache_dir} \
            --output_dir ${output_dir} \
            --logging_dir ${logging_dir} \
            --preprocessing_num_workers ${num_workers} \
            --dataloader_num_workers ${num_workers} \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size ${per_device_eval_batch_size} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${lr} \
            --num_train_epochs ${num_train_epochs} \
            --optim adafactor \
            --lr_scheduler_type constant \
            --gradient_checkpointing

        # inference
        test_file="data/${task_name}/test.json"
        gen_output_dir="${output_dir}/gen"

        python -m torch.distributed.launch --master_port ${master_port} \
            --nproc_per_node ${n_gpus} ../run_seq2seq.py \
            --task_name ${task_name} \
            --dataset_name ${dataset_path} \
            --dataset_config_name ${task_name} \
            --metric_name_or_path ${metric_name_or_path} \
            --metric_config_name ${task_name} \
            --test_file ${test_file} \
            --source_column ${source_column} \
            --target_column ${target_column} \
            --max_source_length ${max_source_length} \
            --max_target_length ${max_target_length} \
            --truncation_side ${truncation_side} \
            --model_name_or_path ${output_dir} \
            --do_predict \
            --predict_with_generate \
            --cache_dir ${cache_dir} \
            --output_dir ${gen_output_dir} \
            --logging_dir ${logging_dir} \
            --overwrite_output_dir \
            --preprocessing_num_workers ${num_workers} \
            --dataloader_num_workers ${num_workers} \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size ${per_device_eval_batch_size} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${lr} \
            --num_train_epochs ${num_train_epochs} \
            --optim adafactor \
            --lr_scheduler_type constant \
            --gradient_checkpointing
        
    done
done

# evaluation
python evaluate.py --output_dirs output/${model_name} -t ${task_name} -s 50 100 200 -o 0 1 2 3 4
