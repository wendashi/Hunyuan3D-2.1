#!/bin/bash
# 第一阶段训练脚本
# 训练 1-8 parts 模型

set -e

# 配置参数
CONFIG="/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/configs/ours_hunyuandit-finetuning-flowmatching-dinog518-bf16-lr1e5-4096.yaml"
# 从配置文件中读取 batch_size
BS=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG')); print(config['dataset']['params']['batch_size'])")
OUTPUT_DIR="/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit+PC/minimal_bs=$BS"
NUM_NODES=1
UPDATE_EVERY=2 # 梯度累积步数
CUDA_DEVICES="0,3"  # 指定使用的 GPU
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# 可选：从 checkpoint 恢复训练
# CKPT_PATH="./outputs/stage1_test/ckpt/ckpt-step=00001000.ckpt"

echo "🚀 开始第一阶段训练 (1-$BS parts)"
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BS (从配置文件读取)"
echo "GPU 数量: $NUM_GPUS"
echo "梯度累积: $UPDATE_EVERY"
echo "使用 GPU: $CUDA_DEVICES"

# 设置环境变量
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# 训练命令
python /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/main.py \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed2 \
    --update_every "$UPDATE_EVERY"
    # --ckpt_path "$CKPT_PATH"  # 取消注释以从 checkpoint 恢复

echo "✅ 第一阶段训练完成"
echo "输出目录: $OUTPUT_DIR"
