#!/bin/bash
# 第一阶段训练脚本
# 训练 1-8 parts 模型

set -e

# 配置参数
CONFIG="/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/configs/hunyuandit-mini-overfitting-flowmatching-dinog518-bf16-lr1e4-4096.yaml"
# 从配置文件中读取 batch_size
BS=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG')); print(config['dataset']['params']['batch_size'])")
NUM_LATENTS=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG')); print(config['model']['params']['first_stage_config']['params']['num_latents'])")
OUTPUT_DIR="/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-minimal-overfitting/whole_bs=${BS}_latents=${NUM_LATENTS}"
NUM_NODES=1
UPDATE_EVERY=2 # 梯度累积步数
CUDA_DEVICES="6,7"  # 指定使用的 GPU
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# 可选：从 checkpoint 恢复训练
# 想在不同 GPU 数量上继续训练：把脚本里的 CKPT_PATH 改成 .../ckpt-step-xxxx.fp32，运行时会自动载入模型参数并冻结不需要训练的分支。
# 想按原 ZeRO 检查点完整续训（需匹配 GPU 数量）：把 CKPT_PATH 指向 ckpt-step-xxxx.ckpt 目录即可，逻辑会自动绕过手动加载。
CKPT_PATH="${OUTPUT_DIR}/ckpt/ckpt-step=00002000-v1.ckpt/ckpt-step=00002000-v1.fp32" # 转换后的checkpoint

echo "🚀 开始第一阶段训练 (1-$BS parts)"
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BS (从配置文件读取)"
echo "GPU 数量: $NUM_GPUS"
echo "梯度累积: $UPDATE_EVERY"
echo "使用 GPU: $CUDA_DEVICES"

# 设置环境变量, 解决NCCL超时问题
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export NCCL_BLOCKING_WAIT=1          # 或 TORCH_NCCL_BLOCKING_WAIT=1，二选一即可
export NCCL_ASYNC_ERROR_HANDLING=1   # 建议同时打开，出错能及时退出
export NCCL_TIMEOUT=1800             # 单位秒，可根据需要调大到 1800、2400 等

# 训练命令
python /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/main_ours.py \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed2 \
    --update_every "$UPDATE_EVERY" \
    # --ckpt_path "$CKPT_PATH"  # 取消注释以从 checkpoint 恢复

echo "✅ 第一阶段训练完成"
echo "输出目录: $OUTPUT_DIR"
