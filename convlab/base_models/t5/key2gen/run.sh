for ratio in 0.1 0.01
do
    for dial_ids_order in 0 1 2
    do
        bash run_persona_fewshot_key2gen.sh ${ratio} ${dial_ids_order}
    done
done