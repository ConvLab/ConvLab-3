task_name="lm"
model_type="gpt"
model_name_or_path="gpt2-large"
keywords_num=100
keywords_ratio=0.3
keywords_th_ratio=0
stopwords=True
for dataset_name in dailydialog metalwoz tm1 tm2 tm3 sgd reddit wikidialog
do
    data_dir="data/${task_name}/${model_type}/${dataset_name}"
    for data_split in validation train
    do
        token_loss_file="${data_dir}/token_loss_${data_split}.json"
        output_file="${data_dir}/keywords_${data_split}.json"
        python lmloss2keywords.py \
            --model_type ${model_type} \
            --model_name_or_path ${model_name_or_path} \
            --token_loss_file ${token_loss_file} \
            --keywords_num ${keywords_num} \
            --keywords_ratio ${keywords_ratio} \
            --keywords_th_ratio ${keywords_th_ratio} \
            --stopwords ${stopwords} \
            --output_file ${output_file}
    done
done