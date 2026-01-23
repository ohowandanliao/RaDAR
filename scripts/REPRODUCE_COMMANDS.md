# RaDAR SOTA 复现命令汇总

本文档包含论文中所有可复现的SOTA实验命令，基于日志文件提取的参数。

## 环境设置

```bash
# Conda环境
source /data/yixuan/anaconda3/etc/profile.d/conda.sh
conda activate /data/yixuan/anaconda3/envs/radar
cd /home2/yixuan/Radar

# 可用GPU: 4, 5, 6, 7 (A800-SXM4-80GB)
```

## 1. Binary-Edge 数据集 (Table 5)

### 1.1 Last.FM
**论文结果**: Recall@20=0.2724, NDCG@20=0.1992, Recall@40=0.3664, NDCG@40=0.2309

```bash
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
    --exp sota_lastfm_reproduce
```

### 1.2 Yelp
**论文结果**: Recall@20=0.0914, NDCG@20=0.0464, Recall@40=0.1355, NDCG@40=0.0571

```bash
CUDA_VISIBLE_DEVICES=5 python Main.py \
    --data yelp \
    --epoch 250 \
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
    --use_diff_gcl 0 \
    --exp sota_yelp_reproduce
```

### 1.3 BeerAdvocate
**论文结果**: Recall@20=0.1273, NDCG@20=0.1061, Recall@40=0.1942, NDCG@40=0.1375

```bash
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
    --exp sota_beer_reproduce
```

## 2. Weighted-Edge 数据集 (Table 6)

### 2.1 Tmall (主变体)
**论文结果**: Recall@20=0.0626, NDCG@20=0.0268

```bash
CUDA_VISIBLE_DEVICES=4 python Main.py \
    --data tmall \
    --epoch 200 \
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
    --exp tmall_weighted_gcl
```

### 2.2 Tmall (+DDR变体)
**论文结果**: Recall@20=0.0620, NDCG@20=0.0260

```bash
CUDA_VISIBLE_DEVICES=5 python Main.py \
    --data tmall \
    --epoch 200 \
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
    --exp tmall_weighted_gcl_ddr
```

### 2.3 RetailRocket (主变体)
**论文结果**: Recall@20=0.1380, NDCG@20=0.0746

```bash
CUDA_VISIBLE_DEVICES=6 python Main.py \
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
    --exp rr_weighted_gcl
```

### 2.4 RetailRocket (+DDR变体)
**论文结果**: Recall@20=0.1375, NDCG@20=0.0748

```bash
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
    --exp rr_weighted_gcl_ddr
```

### 2.5 IJCAI15 (主变体)
**论文结果**: Recall@20=0.0582, NDCG@20=0.0323

```bash
CUDA_VISIBLE_DEVICES=4 python Main.py \
    --data ijcai_15 \
    --epoch 200 \
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
    --exp ijcai15_weighted_gcl
```

### 2.6 IJCAI15 (+DDR变体)
**论文结果**: Recall@20=0.0603, NDCG@20=0.0325

```bash
CUDA_VISIBLE_DEVICES=5 python Main.py \
    --data ijcai_15 \
    --epoch 200 \
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
    --exp ijcai15_weighted_gcl_ddr
```

## 3. 快速启动脚本

### 运行所有实验
```bash
bash /home2/yixuan/Radar/scripts/run_all_reproduce.sh all
```

### 仅运行Binary-edge实验
```bash
bash /home2/yixuan/Radar/scripts/run_all_reproduce.sh binary
```

### 仅运行Weighted-edge实验
```bash
bash /home2/yixuan/Radar/scripts/run_all_reproduce.sh weighted
```

## 4. 结果验证清单

| 数据集 | 类型 | Recall@20 | NDCG@20 | 日志路径 |
|--------|------|-----------|---------|----------|
| Last.FM | Binary | 0.2724 | 0.1992 | result/sota/lastfm/ |
| Yelp | Binary | 0.0914 | 0.0464 | result/sota/yelp/ |
| Beer | Binary | 0.1273 | 0.1061 | result/sota/beer/ |
| Tmall | Weighted | 0.0626 | 0.0268 | result/tmall_weighted_gcl/ |
| Tmall+DDR | Weighted | 0.0620 | 0.0260 | result/tmall_weighted_gcl_ddr/ |
| RetailRocket | Weighted | 0.1380 | 0.0746 | result/rr_weighted_gcl/ |
| RetailRocket+DDR | Weighted | 0.1375 | 0.0748 | result/rr_weighted_gcl_ddr/ |
| IJCAI15 | Weighted | 0.0582 | 0.0323 | result/ijcai15_weighted_gcl/ |
| IJCAI15+DDR | Weighted | 0.0603 | 0.0325 | result/ijcai15_weighted_gcl_ddr/ |

## 5. 关键参数说明

| 参数 | Binary-edge | Weighted-edge | 说明 |
|------|-------------|---------------|------|
| `--cl_type` | acl | gcl | ACL用于binary，GCL用于weighted |
| `--use_diff_gcl` | 0/1 | 0/1 | 是否启用diffusion |
| `--use_weighted_edges` | 0 | 1 | 是否使用加权边 |
| `--acl_ratio` | 2.5-5.5 | 1.0 | ACL权重比例 |
| `--lambda_ddr` | - | 0.03 | DDR正则化权重 |
| `--ddr_warmup` | - | 30 | DDR预热轮数 |
