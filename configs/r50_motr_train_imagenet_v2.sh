#!/usr/bin/env bash
#SBATCH --partition=4090
#SBATCH --nodelist=3dimage-17
#SBATCH --gres=gpu:8
#一个epoch太长了，改成一半datalodaer，提前看看
# for MOT17
# upmotr
EXP_DIR=exps/e2e_motr_r50_imagenet_v2
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --meta_arch motr_pretrain \
    --use_checkpoint \
    --dataset_file e2e_imagenet \
    --epoch 1 \
    --with_box_refine \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --iou 0.1 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --mot_path /space/mawb/MOTR/data/Dataset/mot \
    --fre_cnn \
    --num_pictures_row 2 \
    --num_pictures_column 3 \
    --num_queries 300 \
    |& tee ${EXP_DIR}/output.log
