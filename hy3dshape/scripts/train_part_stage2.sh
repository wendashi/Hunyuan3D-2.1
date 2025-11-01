#!/bin/bash
# 第二阶段训练脚本
# 基于第一阶段模型，训练 1-16 parts

set -e

CONFIG="configs/ours_multipart_stage2.yaml"
OUTPUT_DIR="./outputs/stage2"
STAGE1_OUTPUT="./outputs/stage1"

echo "🚀 开始第二阶段训练 (1-16 parts)"
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"

# 查找第一阶段的最新检查点
STAGE1_CHECKPOINT=$(find "$STAGE1_OUTPUT/checkpoints" -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$STAGE1_CHECKPOINT" ]; then
    echo "⚠️  未找到第一阶段检查点，将从头开始训练"
    python train.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    echo "📁 找到第一阶段检查点: $STAGE1_CHECKPOINT"
    python train.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --resume_from_checkpoint "$STAGE1_CHECKPOINT"
fi

echo "✅ 第二阶段训练完成"
echo "输出目录: $OUTPUT_DIR"
