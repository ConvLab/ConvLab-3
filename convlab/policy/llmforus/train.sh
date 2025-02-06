

expdir="exp/promptEmoUS-2023-06-22"
model_path="../test-llm/7B" # the path for llama model
llmforus_data="convlab/policy/llmforus/unify/data"
trainfile="${llmforus_data}/emowoz+dialmage/train.json"
valfile="${llmforus_data}/emowoz+dialmage/validation.json"

mkdir -p $expdir
accelerate launch accel_train_model.py \
    --model_path $model_path \
    --batch_size 2 \
    --eval_batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 2000 \
    --train_data_path $trainfile \
    --topn 1 \
    --val_data_path $valfile 
    # --resume exp/slurp_llama13b_baseline_2000_samples \
    # --tag noschema \
    # --ontology dataset/KB.json \
    # --maxKBsize 10 \
    # --KBdrop 0.5 \

# gsutil -m cp -r $expdir gs://hsienchin/emoUS/
# python3 emotion_evaluation.py --data dataset/emowoz+dialmage/test.json --model-checkpoint ../test-llm/7B --peft-checkpoint $expdir
# gsutil -m cp -r $expdir/result gs://hsienchin/emoUS/$expdir/
# sudo poweroff
