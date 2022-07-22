set -e
n_gpus=2
task_name="rg"
dataset_name="metalwoz+sgd+tm1+tm2+tm3"
speaker="all"
data_dir="data/${task_name}/${dataset_name}/${speaker}"
output_dir="output/${task_name}/${dataset_name}/${speaker}"
cache_dir="../cache"
logging_dir="${output_dir}/runs"
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
source_column="context"
target_column="response"
truncation_side="left"
max_source_length=512
max_target_length=128
model_name_or_path="t5-small"
per_device_train_batch_size=128
per_device_eval_batch_size=128
gradient_accumulation_steps=4
lr=1e-3
num_train_epochs=1

names=$(echo ${dataset_name} | tr "+" "\n")
rm -r ${data_dir}
mkdir -p ${data_dir}
for name in ${names};
do
    echo "preprocessing ${name}"
    python ../create_data.py --tasks ${task_name} --datasets ${name} --speaker ${speaker}
    if [ "${name}" != "${dataset_name}" ]; then
        cat "data/${task_name}/${name}/${speaker}/train.json" >> ${train_file}
        cat "data/${task_name}/${name}/${speaker}/validation.json" >> ${validation_file}
        cat "data/${task_name}/${name}/${speaker}/test.json" >> ${test_file}
    fi
done

python -m torch.distributed.launch \
    --nproc_per_node ${n_gpus} ../run_seq2seq.py \
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
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
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
