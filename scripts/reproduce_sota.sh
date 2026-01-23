#!/bin/bash
# =============================================================================
# RaDAR SOTA Reproduction Script
# Binary-edge datasets: lastfm, yelp, beer
# =============================================================================

set -euo pipefail

CONDA_ENV="/data/yixuan/anaconda3/envs/radar"
WORK_DIR="/home2/yixuan/Radar"

source /data/yixuan/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $WORK_DIR

echo "=================================================="
echo "RaDAR SOTA Reproduction - Binary Edge Datasets"
echo "=================================================="

# =============================================================================
# Last.FM: R@20=0.2724, N@20=0.1992
# =============================================================================
echo "[1/3] Last.FM - GPU 4"

CUDA_VISIBLE_DEVICES=4 python Main.py \
    --data lastfm \
    --epoch 400 \
    --latdim 64 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --ssl_reg 0.1 \
    --ib_reg 0.1 \
    --temp 0.5 \
    --gamma -0.95 \
    --lambda0 0.0001 \
    --acl_ratio 5.5 \
    --acl_mlp_nums 2 \
    --attention_type gate \
    --cl_type acl \
    --use_diff_gcl 1 \
    --noise_scale 0.1 \
    --noise_min 0.0001 \
    --noise_max 0.02 \
    --diff_steps 5 \
    --d_emb_size 10 \
    --save_best 1 \
    --exp reproduce_binary &

LASTFM_PID=$!

# =============================================================================
# Yelp: R@20=0.0914, N@20=0.0464
# =============================================================================
echo "[2/3] Yelp - GPU 5"

CUDA_VISIBLE_DEVICES=5 python Main.py \
    --data yelp \
    --epoch 150 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --ssl_reg 1.0 \
    --ib_reg 0.1 \
    --temp 0.5 \
    --gamma -0.45 \
    --lambda0 0.0001 \
    --acl_ratio 2.5 \
    --acl_mlp_nums 2 \
    --attention_type gate \
    --cl_type acl \
    --use_diff_gcl 1 \
    --noise_scale 0.1 \
    --noise_min 0.0001 \
    --noise_max 0.02 \
    --diff_steps 5 \
    --d_emb_size 10 \
    --save_best 1 \
    --exp reproduce_binary &

YELP_PID=$!

# =============================================================================
# BeerAdvocate: R@20=0.1273, N@20=0.1061
# =============================================================================
echo "[3/3] BeerAdvocate - GPU 6"

CUDA_VISIBLE_DEVICES=6 python Main.py \
    --data beer \
    --epoch 180 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --ssl_reg 1.0 \
    --ib_reg 0.01 \
    --temp 0.5 \
    --gamma -0.45 \
    --lambda0 0.01 \
    --acl_ratio 2.5 \
    --acl_mlp_nums 2 \
    --attention_type gate \
    --cl_type acl \
    --use_diff_gcl 1 \
    --noise_scale 0.1 \
    --noise_min 0.0001 \
    --noise_max 0.02 \
    --diff_steps 5 \
    --d_emb_size 10 \
    --save_best 1 \
    --exp reproduce_binary &

BEER_PID=$!

echo ""
echo "=================================================="
echo "All 3 binary-edge experiments started"
echo "PIDs: lastfm=$LASTFM_PID, yelp=$YELP_PID, beer=$BEER_PID"
echo "Results -> result/reproduce_binary/"
echo "=================================================="

wait $LASTFM_PID $YELP_PID $BEER_PID

echo "All binary-edge experiments completed!"
