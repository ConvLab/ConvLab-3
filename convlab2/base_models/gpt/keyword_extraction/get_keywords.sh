task_name="lm"
dataset_name=$1
model_type="gpt"
data_dir="data/${task_name}/${dataset_name}/${model_type}"
model_name_or_path="gpt2-large"
keywords_num=100
keywords_ratio=0.3
keywords_th_ratio=0
stopwords=True
for data_split in validation test train
do
    word_loss_file="${data_dir}/${model_name_or_path}_${dataset_name}_${data_split}_word_loss.json"
    output_file="${data_dir}/${dataset_name}_${data_split}_keywords_${model_name_or_path}_topk_${keywords_num}_ratio_${keywords_ratio}_th_${keywords_th_ratio}_stopwords_${stopwords}.json"

    python lmloss2keywords.py \
        --model_type ${model_type} \
        --word_loss_file ${word_loss_file} \
        --keywords_num ${keywords_num} \
        --keywords_ratio ${keywords_ratio} \
        --keywords_th_ratio ${keywords_th_ratio} \
        --stopwords ${stopwords} \
        --output_file ${output_file}
done