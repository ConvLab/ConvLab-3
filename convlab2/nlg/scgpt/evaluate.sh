CUDA_VISIBLE_DEVICES="5" python -m torch.distributed.launch --nproc_per_node 1 --master_port 3046 main.py \
--dataset multiwoz21 \
--scgpt_model_ckpt_path /data/zhangzheng/scgpt \
--model_path /data/zhangzheng/ConvLab-3/convlab2/nlg/scgpt/saved_model/epoch_4/epoch_4_step8875.pt