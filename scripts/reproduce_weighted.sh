#!/bin/bash
# =============================================================================
# RaDAR SOTA Reproduction Script - Weighted Edge Datasets
# Datasets: Tmall, RetailRocket, IJCAI15 (RetailRocket -> IJCAI15 on GPUs 3/7)
# GPUs: 0, 1, 3, 7 (load balanced)
# =============================================================================

set -euo pipefail

CONDA_ENV="/data/yixuan/anaconda3/envs/radar"
WORK_DIR="/home2/yixuan/Radar"

source /data/yixuan/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
cd $WORK_DIR

echo "=================================================="
echo "RaDAR SOTA Reproduction - Weighted Edge Datasets"
echo "GPUs: 0, 1, 3, 7"
echo "=================================================="

# =============================================================================
# GPU 0: Tmall main (100)
# =============================================================================
(
echo "[GPU 0] Tmall (main)"

CUDA_VISIBLE_DEVICES=0 python Main.py \
    --data tmall \
    --epoch 100 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.35 \
    --ssl_reg 0.20 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 0 \
    --save_best 1 \
    --exp reproduce_weighted
) &
GPU0_PID=$!

# =============================================================================
# GPU 1: Tmall DDR (100)
# =============================================================================
(
echo "[GPU 1] Tmall (+DDR)"

CUDA_VISIBLE_DEVICES=1 python Main.py \
    --data tmall \
    --epoch 100 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.35 \
    --ssl_reg 0.20 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 1 \
    --lambda_ddr 0.03 \
    --ddr_warmup 30 \
    --save_best 1 \
    --exp reproduce_weighted_ddr
) &
GPU1_PID=$!

# =============================================================================
# GPU 3: RetailRocket main (200) -> IJCAI15 main (85)
# =============================================================================
(
echo "[GPU 3] RetailRocket (main) -> IJCAI15 (main)"

CUDA_VISIBLE_DEVICES=3 python Main.py \
    --data retail_rocket \
    --epoch 200 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.40 \
    --ssl_reg 0.15 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 0 \
    --save_best 1 \
    --exp reproduce_weighted

CUDA_VISIBLE_DEVICES=3 python Main.py \
    --data ijcai_15 \
    --epoch 85 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.35 \
    --ssl_reg 0.20 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 0 \
    --save_best 1 \
    --exp reproduce_weighted
) &

GPU3_PID=$!

# =============================================================================
# GPU 7: RetailRocket DDR (200) -> IJCAI15 DDR (85)
# =============================================================================
(
echo "[GPU 7] RetailRocket (+DDR) -> IJCAI15 (+DDR)"

CUDA_VISIBLE_DEVICES=7 python Main.py \
    --data retail_rocket \
    --epoch 200 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.40 \
    --ssl_reg 0.15 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 1 \
    --lambda_ddr 0.03 \
    --ddr_warmup 30 \
    --save_best 1 \
    --exp reproduce_weighted_ddr

CUDA_VISIBLE_DEVICES=7 python Main.py \
    --data ijcai_15 \
    --epoch 85 \
    --latdim 128 \
    --gnn_layer 2 \
    --lr 0.001 \
    --batch 4096 \
    --reg 1e-5 \
    --temp 0.35 \
    --ssl_reg 0.20 \
    --ib_reg 0.05 \
    --attention_type gate \
    --cl_type gcl \
    --use_weighted_edges 1 \
    --use_diff_gcl 1 \
    --lambda_ddr 0.03 \
    --ddr_warmup 30 \
    --save_best 1 \
    --exp reproduce_weighted_ddr
) &

GPU7_PID=$!

echo ""
echo "=================================================="
echo "All weighted-edge experiments started"
echo "GPU 0: Tmall (main)"
echo "GPU 1: Tmall (+DDR)"
echo "GPU 3: RetailRocket -> IJCAI15 (main)"
echo "GPU 7: RetailRocket -> IJCAI15 (+DDR)"
echo "Results -> result/reproduce_weighted/ & result/reproduce_weighted_ddr/"
echo "=================================================="

wait $GPU0_PID $GPU1_PID $GPU3_PID $GPU7_PID

echo "All weighted-edge experiments completed!"
