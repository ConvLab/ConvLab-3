set -e

dataset_names=(multiwoz21 sgd)
src_data_dirs=(data/multiwoz21 data/sgd/group0)
aug_types=(0   1   2   3   4   5   4     8     8   5   5   11  12  13  14  15  16  17  18  19)
aug_times=(0.0 0.1 2.0 2.0 2.0 2.0 100.0 100.0 2.0 4.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0)
model_types=(0 1 3 4 5 7)
context_window_sizes=(100 4 100 100 100 100)

qadst_dir=$1
i=$2
j=$3
k=$4
master_port=$5
model_size=t5-large
dataset_name=${dataset_names[i]}
src_data_dir=${src_data_dirs[i]}
aug_type=${aug_types[j]}
aug_time=${aug_times[j]}
model_type=${model_types[k]}
context_window_size=${context_window_sizes[k]}
read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
if ((${i}==0))
then
    output_dir="output0615/${dataset_name}/${qadst_dir}/${model_size}_aug${aug_type}_x${aug_time}_model${model_type}_context${context_window_size}"
else
    output_dir="output0615/${dataset_name}/group0/${qadst_dir}/${model_size}_aug${aug_type}_x${aug_time}_model${model_type}_context${context_window_size}"
fi

bash run_dst.sh ${dataset_name} ${model_size} ${model_type} ${src_data_dir} ${read_data_dir} ${output_dir} ${context_window_size} ${master_port}
