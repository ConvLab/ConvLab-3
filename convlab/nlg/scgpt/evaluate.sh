CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 2052 main.py \
--batch_size 1 \
--base_model_name_path gpt2-medium \
--dataset tm3 \
--exp_name tm3_mst_test \
--model_path saved_models/mwoz_sgd_tm_train/epoch_5/epoch_5_step19206.pt \
# --model_path saved_models/gpt2_tm_direct/epoch_19/epoch_19_step65540.pt \
# --model_path saved_models/gpt2_tm_direct/epoch_6/epoch_6_step22939.pt \