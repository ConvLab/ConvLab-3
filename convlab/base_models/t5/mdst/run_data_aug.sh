set -e

qadst_dir="qadst_f1_th>0.1"
dataset_names=(multiwoz21 sgd)
src_data_dirs=(data/multiwoz21 data/sgd/group0)
aug_types=(0   1   2   3   4   5   4     8     8   5   5   11  12  13  14  15  16  17  18  19)
aug_times=(0.0 0.1 2.0 2.0 2.0 2.0 100.0 100.0 2.0 4.0 1.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0)
model_types=(0 1 3 4 5 7)
context_window_sizes=(100 4 100 100 100 100)

# data for rewrite (aug4_x100)
for i in 0 1
do
    dataset_name=${dataset_names[i]}
    src_data_dir=${src_data_dirs[i]}
    for j in 6
    do
        aug_type=${aug_types[j]}
        aug_time=${aug_times[j]}
        read_data_dir="${src_data_dir}/${qadst_dir}"
        write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

        if ((${i}==0))
        then
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
            python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
        else
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
            python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
        fi
        bash run_coqr_syn.sh ${dataset_name} 23457 ${write_data_dir}
    done
done


# Main exp
for i in 0 1
do
    dataset_name=${dataset_names[i]}
    src_data_dir=${src_data_dirs[i]}
    for j in 0 2 5
    do
        # data augmentation
        aug_type=${aug_types[j]}
        aug_time=${aug_times[j]}
        read_data_dir="${src_data_dir}/${qadst_dir}"
        write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

        if ((${i}==0))
        then
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
        else
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
        fi

        # CoQR
        if ((${j}==5))
        then
            if ((${i}==0))
            then
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
            else
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
            fi
        fi

        # domain classifier
        read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
        write_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}/dom_cls_c4"
        
        if ((${i}==0))
        then
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4
        else
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4 -g 0
        fi

        # DST models
        for k in 0 1 2 3 4 5
        do
            model_type=${model_types[k]}
            context_window_size=${context_window_sizes[k]}
            write_data_dir="${read_data_dir}/model${model_type}_context${context_window_size}"
            if ((${i}==0))
            then
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir}
            else
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir} -g 0
            fi
        done
    done
done


# partial SYN exp
for i in 1
do
    dataset_name=${dataset_names[i]}
    src_data_dir=${src_data_dirs[i]}
    for j in 18 19
    do
        # data augmentation
        aug_type=${aug_types[j]}
        aug_time=${aug_times[j]}
        read_data_dir="${src_data_dir}/${qadst_dir}"
        write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

        if ((${i}==0))
        then
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
        else
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
        fi

        # CoQR
        if ((${j}==5))
        then
            if ((${i}==0))
            then
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
            else
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
            fi
        fi

        # domain classifier
        read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
        write_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}/dom_cls_c4"
        
        if ((${i}==0))
        then
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4
        else
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4 -g 0
        fi

        # DST models
        for k in 0 3
        do
            model_type=${model_types[k]}
            context_window_size=${context_window_sizes[k]}
            write_data_dir="${read_data_dir}/model${model_type}_context${context_window_size}"
            if ((${i}==0))
            then
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir}
            else
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir} -g 0
            fi
        done
    done
done


# aug_times and ablation exp
for i in 0 1
do
    dataset_name=${dataset_names[i]}
    src_data_dir=${src_data_dirs[i]}
    for j in 1 4 9 10
    do
        # data augmentation
        aug_type=${aug_types[j]}
        aug_time=${aug_times[j]}
        read_data_dir="${src_data_dir}/${qadst_dir}"
        write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

        if ((${i}==0))
        then
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
        else
            python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
        fi

        # CoQR
        if ((${j}==5))
        then
            if ((${i}==0))
            then
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
            else
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
            fi
        fi

        # domain classifier
        read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
        write_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}/dom_cls_c4"
        
        if ((${i}==0))
        then
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4
        else
            python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4 -g 0
        fi

        # DST models
        for k in 3
        do
            model_type=${model_types[k]}
            context_window_size=${context_window_sizes[k]}
            write_data_dir="${read_data_dir}/model${model_type}_context${context_window_size}"
            if ((${i}==0))
            then
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir}
            else
                python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir} -g 0
            fi
        done
    done
done


for qadst_dir in "qadst_f1_th>0" "qadst_f1_th>0.3" "qadst_f1_th>0.5" "qadst_f1_th>0.8""qadst_true_slot_pairs"
do
    for i in 0 1
    do
        dataset_name=${dataset_names[i]}
        src_data_dir=${src_data_dirs[i]}
        for j in 6
        do
            aug_type=${aug_types[j]}
            aug_time=${aug_times[j]}
            read_data_dir="${src_data_dir}/${qadst_dir}"
            write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

            if ((${i}==0))
            then
                python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
            else
                python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
                python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
            fi
            bash run_coqr_syn.sh ${dataset_name} 23457 ${write_data_dir}
        done
        for j in 5
        do
            # data augmentation
            aug_type=${aug_types[j]}
            aug_time=${aug_times[j]}
            read_data_dir="${src_data_dir}/${qadst_dir}"
            write_data_dir="${read_data_dir}/aug${aug_type}_x${aug_time}"

            if ((${i}==0))
            then
                python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time}
            else
                python create_data.py -t aug_data -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -a ${aug_type} -x ${aug_time} -g 0
            fi

            # CoQR
            if ((${j}==5))
            then
                if ((${i}==0))
                then
                    python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type}
                else
                    python create_data.py -t coqr -d ${dataset_name} -w ${write_data_dir} -a ${aug_type} -g 0
                fi
            fi

            # domain classifier
            read_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}"
            write_data_dir="${src_data_dir}/${qadst_dir}/aug${aug_type}_x${aug_time}/dom_cls_c4"
            
            if ((${i}==0))
            then
                python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4
            else
                python create_data.py -t dom_cls -d ${dataset_name} -r ${read_data_dir} -w ${write_data_dir} -c 4 -g 0
            fi

            # DST models
            for k in 3
            do
                model_type=${model_types[k]}
                context_window_size=${context_window_sizes[k]}
                write_data_dir="${read_data_dir}/model${model_type}_context${context_window_size}"
                if ((${i}==0))
                then
                    python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir}
                else
                    python create_data.py -t dst -d ${dataset_name} -m ${model_type} -c ${context_window_size} -r ${read_data_dir} -w ${write_data_dir} -g 0
                fi
            done
        done
    done
done