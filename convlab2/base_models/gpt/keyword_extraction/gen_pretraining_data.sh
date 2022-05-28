task_name="key2gen_noisy"
dataset_name="dailydialog+metalwoz+tm1+tm2+tm3"
names=$(echo ${dataset_name} | tr "+" "\n")
model_type="gpt"
data_dir=data/${task_name}/${model_type}/${name}/${dataset_name}
rm -r ${data_dir}
mkdir -p ${data_dir}
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
for name in ${names}
do
    echo "preprocessing ${name}"
    python gen_pretraining_data.py -i data/lm/${name}/${model_type} -o data/${task_name}/${model_type}/${name} -m ${task_name}
    if [ "${name}" != "${dataset_name}" ]; then
        cat "data/${task_name}/gpt/${name}/train.json" >> ${train_file}
        cat "data/${task_name}/gpt/${name}/validation.json" >> ${validation_file}
        cat "data/${task_name}/gpt/${name}/test.json" >> ${test_file}
    fi
done
python gen_pretraining_data.py -i data/lm/multiwoz21/${model_type} -o data/${task_name}/${model_type}/multiwoz21 -m ${task_name}