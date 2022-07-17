# generate data for response generation, key2gen, key2gen_noisy
for task_name in rg
do
    dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+sgd+reddit+wikidialog"
    names=$(echo ${dataset_name} | tr "+" "\n")
    model_type="gpt"
    data_dir=data/${task_name}/${model_type}/${dataset_name}
    mkdir -p ${data_dir}
    train_file="${data_dir}/train.json"
    validation_file="${data_dir}/validation.json"
    test_file="${data_dir}/test.json"
    rm ${train_file} ${validation_file} ${test_file}
    for name in ${names}
    do
        echo "preprocessing ${name}"
        python gen_pretraining_data.py -i data/lm/${model_type}/${name} -o data/${task_name}/${model_type}/${name} -m ${task_name}
        if [ "${name}" != "${dataset_name}" ]; then
            cat "data/${task_name}/${model_type}/${name}/train.json" >> ${train_file}
            cat "data/${task_name}/${model_type}/${name}/validation.json" >> ${validation_file}
            cat "data/${task_name}/${model_type}/${name}/test.json" >> ${test_file}
        fi
    done
done


# # generate data for sentence grounded generation
# task_name="key2gen"
# dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+wikidialog"
# names=$(echo ${dataset_name} | tr "+" "\n")
# model_type="gpt"
# data_dir=data/${task_name}/${model_type}/${dataset_name}
# mkdir -p ${data_dir}
# n_splits=2
# for ((n=0;n<${n_splits};n++))
# do
#     rm ${data_dir}/train_split_${n}-of-${n_splits}.json ${data_dir}/validation_split_${n}-of-${n_splits}.json ${data_dir}/test_split_${n}-of-${n_splits}.json
# done
# for name in ${names}
# do
#     echo "preprocessing ${name}"
#     python gen_pretraining_data.py -i data/lm/${name}/${model_type} -o data/${task_name}/${model_type}/${name} -m ${task_name} -n ${n_splits}
#     if [ "${name}" != "${dataset_name}" ]; then
#         for ((n=0;n<${n_splits};n++))
#         do
#             cat "data/${task_name}/gpt/${name}/train_split_${n}-of-${n_splits}.json" >> "${data_dir}/train_split_${n}-of-${n_splits}.json"
#             cat "data/${task_name}/gpt/${name}/validation_split_${n}-of-${n_splits}.json" >> "${data_dir}/validation_split_${n}-of-${n_splits}.json"
#             cat "data/${task_name}/gpt/${name}/test_split_${n}-of-${n_splits}.json" >> "${data_dir}/test_split_${n}-of-${n_splits}.json"
#         done
#     fi
# done

# # merge generated data with original data
# task_name="sen2gen"
# dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+wikidialog"
# names=$(echo ${dataset_name} | tr "+" "\n")
# model_type="gpt"
# data_dir=data/${task_name}/${model_type}/${dataset_name}
# mkdir -p ${data_dir}
# python gen_pretraining_data.py -i data/key2gen/${model_type}/${dataset_name} -o data/${task_name}/${model_type}/${dataset_name} -m ${task_name}

# # generate sen2gen_noisy data with original data
# task_name="sen2gen_noisy"
# dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+wikidialog"
# names=$(echo ${dataset_name} | tr "+" "\n")
# model_type="gpt"
# data_dir=data/${task_name}/${model_type}/${dataset_name}
# mkdir -p ${data_dir}
# python gen_pretraining_data.py -i data/sen2gen/${model_type}/${dataset_name} -o data/${task_name}/${model_type}/${dataset_name} -m ${task_name}

# merge data for multitask training
# task_name="rg+key2gen+key2gen_noisy+sen2gen+sen2gen_noisy"
# dataset_name="dailydialog+metalwoz+tm1+tm2+tm3+wikidialog"
# model_type="gpt"
# data_dir=data/${task_name}/${model_type}/${dataset_name}
# mkdir -p ${data_dir}
# python gen_pretraining_data.py -i data/ -o data/${task_name}/${model_type}/${dataset_name} -m multitask
