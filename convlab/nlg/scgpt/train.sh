CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 2040 main.py \
--batch_size 64 \
--accumulation_step 2 \
--epoch_num 20 \
--lr 5e-5 \
--base_model_name_path gpt2-medium \
--val_step 500 \
--exp_name mwoz_sgd_tm_train \
--do_train \
--dataset multiwoz21_sgd_tm1_tm2_tm3 \
--train_ratio 1.0 \
# --scgpt_model_ckpt_path saved_models/gpt2_sgd_tm/epoch_2/epoch_2_step13698.pt
# --base_model_name_path /root/autodl-tmp/ConvLab-3/convlab/nlg/scgpt/resource/scgpt \
