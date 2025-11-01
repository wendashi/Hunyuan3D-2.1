# 分阶段训练说明

## 概述

采用 PartCrafter 的分阶段训练策略，通过切换不同的数据文件和配置文件实现分阶段训练。

## 训练阶段

### 第一阶段：1-8 parts 模型
- **数据文件**: `merged-train-stage1-8.json` (2863 个样本, 95.4%)
- **配置文件**: `configs/ours_multipart_stage1.yaml`
- **参数**: `batch_size=8`, `lr=1e-4`, `5K iterations`
- **目标**: 训练基础的多 part 模型

### 第二阶段：1-16 parts 模型
- **数据文件**: `merged-train-stage2-16.json` (2983 个样本, 99.4%)
- **配置文件**: `configs/ours_multipart_stage2.yaml`
- **参数**: `batch_size=16`, `lr=5e-5`, `5K iterations`
- **目标**: 基于第一阶段模型，扩展到更多 parts

## 使用方法

### 方法1：完整分阶段训练
```bash
# 运行完整的分阶段训练
./train_stages.sh
```

### 方法2：单独训练各阶段
```bash
# 第一阶段训练
./train_stage1.sh

# 第二阶段训练（会自动从第一阶段检查点继续）
./train_stage2.sh
```

### 方法3：手动训练
```bash
# 第一阶段
python train.py \
    --config configs/ours_multipart_stage1.yaml \
    --output_dir ./outputs/stage1

# 第二阶段（从第一阶段检查点继续）
python train.py \
    --config configs/ours_multipart_stage2.yaml \
    --output_dir ./outputs/stage2 \
    --resume_from_checkpoint ./outputs/stage1/checkpoints/latest.ckpt
```

## 数据文件说明

- `merged-train-stage1-8.json`: 包含 1-8 parts 的样本
- `merged-train-stage2-16.json`: 包含 1-16 parts 的样本
- 高复杂度样本 (>16 parts) 已被过滤掉

## 输出目录

- `./outputs/stage1/`: 第一阶段模型输出
- `./outputs/stage2/`: 第二阶段模型输出

## 优势

1. **遵循 PartCrafter 成功经验**: 分阶段训练策略
2. **数据分布合理**: 第一阶段样本是第二阶段的 23.8 倍
3. **训练稳定**: 避免 loss spikes 和 catastrophic forgetting
4. **内存效率**: 第一阶段使用较小的 batch_size
5. **简单易用**: 使用 Hunyuan3D 原有训练脚本，只需切换配置

## 注意事项

1. 确保数据文件路径正确
2. 第一阶段完成后才能进行第二阶段
3. 第二阶段会自动查找第一阶段的最新检查点
4. 如果第一阶段检查点不存在，第二阶段会从头开始训练
