# generate data for response generation, key2gen, key2gen_noisy
for task_name in rg key2gen key2gen_noisy
do
    dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+sgd+reddit+wikidialog"
    names=$(echo ${dataset_name} | tr "+" "\n")
    model_type="gpt"
    data_dir=data/${task_name}/${model_type}/${dataset_name}
    mkdir -p ${data_dir}
    train_file="${data_dir}/train.json"
    validation_file="${data_dir}/validation.json"
    rm ${train_file} ${validation_file}
    for name in ${names}
    do
        echo "preprocessing ${name}"
        python gen_pretraining_data.py -i data/lm/${model_type}/${name} -o data/${task_name}/${model_type}/${name} -m ${task_name}
        if [ "${name}" != "${dataset_name}" ]; then
            cat "data/${task_name}/${model_type}/${name}/train.json" >> ${train_file}
            cat "data/${task_name}/${model_type}/${name}/validation.json" >> ${validation_file}
        fi
    done
done

# merge key2gen+key2gen_noisy data
task_name="key2gen+key2gen_noisy"
dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+sgd+reddit+wikidialog"
names=$(echo ${task_name} | tr "+" "\n")
model_type="gpt"
data_dir=data/${task_name}/${model_type}/${dataset_name}
mkdir -p ${data_dir}
train_file="${data_dir}/train.json"
validation_file="${data_dir}/validation.json"
rm ${train_file} ${validation_file}
for name in ${names}
do
    echo "preprocessing ${name}"
    if [ "${name}" != "${task_name}" ]; then
        cat "data/${name}/${model_type}/${dataset_name}/train.json" >> ${train_file}
        cat "data/${name}/${model_type}/${dataset_name}/validation.json" >> ${validation_file}
    fi
done