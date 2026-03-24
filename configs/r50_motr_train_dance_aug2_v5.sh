#!/usr/bin/env bash
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-14
#SBATCH --gres=gpu:8

# for MOT17

PRETRAIN=/space/mawb/MOTR/exps/e2e_motr_r50_dance_aug1/checkpoint0019.pth
EXP_DIR=exps/e2e_motr_r50_dance_aug2_v5_test
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --meta_arch motr_dance_aug2_v5 \
    --use_checkpoint \
    --dataset_file e2e_dance_aug2 \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --mot_path /space/mawb/MOTR/data/Dataset/mot
    |& tee ${EXP_DIR}/output.log
