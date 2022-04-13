model_type=dialogpt
dataset_name=multiwoz21
model_name=dialogpt-large
data_dir="data/lm/${dataset_name}/${model_type}"
word_loss_file="${data_dir}/${model_name}_${dataset_name}_word_loss.json"
keywords_num=5
keywords_ratio=1
keywords_th=0
stopwords=True
output_file="${data_dir}/${dataset_name}_keywords_${model_name}_topk_${keywords_num}_ratio_${keywords_ratio}_th_${keywords_th}_stopwords_${stopwords}.json"

python lmloss2keywords.py \
    --model_type ${model_type} \
    --word_loss_file ${word_loss_file} \
    --keywords_num ${keywords_num} \
    --keywords_ratio ${keywords_ratio} \
    --keywords_th ${keywords_th} \
    --stopwords ${stopwords} \
    --output_file ${output_file}
    