dataset_name="metalwoz+sgd+tm1+tm2+tm3"
names=$(echo ${dataset_name} | tr "+" "\n")
model_type="gpt"
data_dir=data/key2gen_shuffle_noisy/${model_type}/${name}/${dataset_name}
mkdir -p ${data_dir}
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
test_file="${data_dir}/test.json"
for name in ${names}
do
    echo "preprocessing ${name}"
    python gen_pretraining_data.py -i data/lm/${name}/${model_type} -o data/key2gen_shuffle_noisy/${model_type}/${name}
    if [ "${name}" != "${dataset_name}" ]; then
        cat "data/key2gen_shuffle_noisy/gpt/${name}/train.json" >> ${train_file}
        cat "data/key2gen_shuffle_noisy/gpt/${name}/validation.json" >> ${validation_file}
        cat "data/key2gen_shuffle_noisy/gpt/${name}/test.json" >> ${test_file}
    fi
done
python gen_pretraining_data.py -i data/lm/multiwoz21/${model_type} -o data/key2gen_shuffle_noisy/${model_type}/multiwoz21