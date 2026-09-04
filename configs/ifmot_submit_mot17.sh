#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${MOT_DATA_ROOT:?Set MOT_DATA_ROOT to the dataset root.}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained IFMOT checkpoint.}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/ifmot_mot17}"
EXP_NAME="${EXP_NAME:-submission}"

python3 submit.py \
    --meta_arch ifmot \
    --dataset_file ifmot_mot17 \
    --mot_path "${MOT_DATA_ROOT}" \
    --with_box_refine \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 10 \
    --sampler_steps 50 90 150 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --resume "${CHECKPOINT}" \
    --exp_name "${EXP_NAME}"
