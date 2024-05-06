set -e

dataset_names=(multiwoz21 sgd)
src_data_dirs=(data/multiwoz21 data/sgd/group0)
aug_types=(0   1   2   3   4   5   4     8     8   5   5   11  12  13  14  15  16  17  18  19)
aug_times=(0.0 0.1 2.0 2.0 2.0 2.0 100.0 100.0 2.0 4.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0)

qadst_dir=$1
i=$2
j=$3
master_port=$4
model_size=t5-large
dataset_name=${dataset_names[i]}
src_data_dir=${src_data_dirs[i]}
aug_type=${aug_types[j]}
aug_time=${aug_times[j]}
read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
context_window_size=4
if ((${i}==0))
then
    output_dir="output0615/${dataset_name}/${qadst_dir}/${model_size}_aug${aug_type}_x${aug_time}_dom_cls_c${context_window_size}"
else
    output_dir="output0615/${dataset_name}/group0/${qadst_dir}/${model_size}_aug${aug_type}_x${aug_time}_dom_cls_c${context_window_size}"
fi

bash run_dom_cls.sh ${dataset_name} ${model_size} ${src_data_dir} ${read_data_dir} ${output_dir} ${context_window_size} ${master_port}
