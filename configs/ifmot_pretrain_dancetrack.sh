#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${MOT_DATA_ROOT:?Set MOT_DATA_ROOT to the dataset root.}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PRETRAINED="${PRETRAINED:-r50_deformable_detr-checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-exps/ifmot_dancetrack_pretrain}"

mkdir -p "${OUTPUT_DIR}"

python3 -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --use_env main.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file ifmot_dancetrack_pretrain \
    --epochs 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained "${PRETRAINED}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --mot_path "${MOT_DATA_ROOT}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
