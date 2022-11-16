CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 2050 main.py \
--batch_size 128 \
--base_model_name_path gpt2-medium \
--dataset sgd \
--exp_name gpt2_sgd_test \
--model_path saved_models/exp_name/epoch_x/epoch_7_step10312.pt \