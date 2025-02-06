set -e

# # Split data
# python split_data.py -d multiwoz21 sgd

# # Train QADST on single-domain dialogs
# bash run_qadst_mwoz.sh 23456
# bash run_qadst_sgd.sh 23456

# # Train CoQR on CANARD
# bash run_coqr_canard.sh 23456

# # Train Value Tagger on Taskmaster
# cd ../../bert
# bash train_bio.sh
# cd ../t5/mdst

# # Data Synthesize
# bash run_data_aug.sh

# # Training DST example
# qadst_dir=("qadst_f1_th>0.1" "qadst_f1_th>0" "qadst_f1_th>0.3" "qadst_f1_th>0.5" "qadst_f1_th>0.8""qadst_true_slot_pairs")
# dataset_names=(multiwoz21 sgd)
# src_data_dirs=(data/multiwoz21 data/sgd/group0)
# output_dirs=(output0615/multiwoz21 output0615/sgd/group0)
# aug_types=(0   1   2   3   4   5   4     8     8   5   5   11  12  13  14  15  16  17  18  19)
# aug_times=(0.0 0.1 2.0 2.0 2.0 2.0 100.0 100.0 2.0 4.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0)
# model_types=(0 1 3 4 5 7)
# context_window_sizes=(100 4 100 100 100 100)
# q=$1
# i=$2
# j=$3
# k=$4
# master_port=$5

# bash run_dom_cls1.sh ${qadst_dir[q]} ${i} ${j} ${master_port}
# bash run_dst1.sh ${qadst_dir[q]} ${i} ${j} ${k} ${master_port}


# # Main exp
# q=0
# for i in 0 1
# do
#     for j in 0 2 5
#     do
#         bash run_dom_cls1.sh ${qadst_dir[q]} ${i} ${j} ${master_port}
#         for k in 0 1 2 3 4 5
#         do
#             bash run_dst1.sh ${qadst_dir[q]} ${i} ${j} ${k} ${master_port}
#         done
#     done
# done

# # partial SYN
# i=1
# k=3
# for j in 18 19
# do
#     bash run_dst1.sh ${qadst_dir[q]} ${i} ${j} ${k} ${master_port}
# done

# # CoQR on-the-fly
# j=2
# dataset_name=${dataset_names[i]}
# src_data_dir=${src_data_dirs[i]}
# read_data_dir=${src_data_dirs[i]}/qadst_f1_th>0.1/aug5_x2.0
# output_dir=${output_dirs[i]}/qadst_f1_th>0.1/t5-large_aug5_x2.0_coqr

# for i in 0 1
# do
#     for k in 3 4 5
#     do
#         model_type=${model_types[k]}
#         context_window_size=${context_window_sizes[k]}
#         bash run_coqr.sh ${dataset_name} ${src_data_dir} ${read_data_dir} ${output_dir} ${model_type} ${context_window_size} ${master_port}
#     done
# done

# # ablation exp
# k=3
# for i in 0 1
# do
#     for j in 1 4 9 10
#     do
#         bash run_dst1.sh ${qadst_dir[q]} ${i} ${j} ${k} ${master_port}
#     done
# done

# # qadst f1 exp
# for q in 1 2 3 4 5
# do
#     for i in 0 1
#     do
#         for j in 5
#         do
#             bash run_dst1.sh ${qadst_dir[q]} ${i} ${j} ${k} ${master_port}
#         done
#     done
# done
