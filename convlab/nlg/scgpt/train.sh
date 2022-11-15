CUDA_VISIBLE_DEVICES="3" python -m torch.distributed.launch --nproc_per_node 1 --master_port 2043 main.py \
--batch_size 32 \
--accumulation_step 4 \
--epoch_num 20 \
--lr 5e-5 \
--base_model_name_path /root/autodl-tmp/ConvLab-3/convlab/nlg/scgpt/resource/scgpt \
--val_step 1000 \
--exp_name scgpt_mwoz \
--do_train \
--dataset sgd \
--train_ratio 1.0 \
# --scgpt_model_ckpt_path saved_models/sgd_tm_1e4/epoch_8/epoch_8_step41094.pt
# --base_model_name_path /root/autodl-tmp/ConvLab-3/convlab/nlg/scgpt/resource/scgpt \
