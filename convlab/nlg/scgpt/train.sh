CUDA_VISIBLE_DEVICES="2" python -m torch.distributed.launch --nproc_per_node 1 --master_port 2042 main.py \
--batch_size 32 \
--accumulation_step 4 \
--epoch_num 100 \
--lr 5e-5 \
--base_model_name_path gpt2-medium \
--val_step 100 \
--exp_name gpt2_mwoz001_direct \
--do_train \
--dataset multiwoz21 \
--train_ratio 0.01 \
# --scgpt_model_ckpt_path saved_models/gpt2_sgd_tm/epoch_2/epoch_2_step13698.pt
# --base_model_name_path /root/autodl-tmp/ConvLab-3/convlab/nlg/scgpt/resource/scgpt \
