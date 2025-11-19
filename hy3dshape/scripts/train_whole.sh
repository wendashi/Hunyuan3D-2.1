#!/bin/bash
# 第一阶段训练脚本
# 训练 1-8 parts 模型

set -e

# 配置参数
CONFIG="/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/configs/ours_wholepart.yaml"
# EPSILON_VALUE 环境变量，如果没有设置则使用默认值 0.25
EPSILON_VALUE=${EPSILON_VALUE:-1}
export EPSILON_VALUE

# 从配置文件中读取 batch_size
BS=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG')); print(config['dataset']['params']['batch_size'])")
# 安全读取 num_latents，如果 first_stage_config 没有 params 字段，则使用默认值 4096
NUM_LATENTS=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG'))
first_stage = config['model']['params'].get('first_stage_config', {})
params = first_stage.get('params', {})
num_latents = params.get('num_latents', 4096)
print(num_latents)
")

# 创建临时配置文件，更新 geo_data_dir_candidates
# 优先级：GEO_DATA_DIR 环境变量 > 配置文件中的 geo_data_dir_candidates > EPSILON_VALUE > 默认值
TEMP_CONFIG=$(mktemp)
GEO_DATA_DIR_USED=$(python3 << EOF
import yaml
import sys
import os

# 读取原始配置
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# 从环境变量获取 GEO_DATA_DIR（最高优先级）
geo_data_dir = os.environ.get('GEO_DATA_DIR')
if geo_data_dir:
    config['dataset']['params']['geo_data_dir_candidates'] = [geo_data_dir]
    print(f"Using GEO_DATA_DIR from environment: {geo_data_dir}", file=sys.stderr)
else:
    # 优先使用配置文件中的原始值
    geo_data_dir_candidates = config['dataset']['params'].get('geo_data_dir_candidates', None)
    if geo_data_dir_candidates and len(geo_data_dir_candidates) > 0:
        geo_data_dir = geo_data_dir_candidates[0]
        print(f"Using geo_data_dir_candidates from config: {geo_data_dir}", file=sys.stderr)
    else:
        # 如果配置文件中没有，再检查 EPSILON_VALUE
        epsilon_value = os.environ.get('EPSILON_VALUE')
        if epsilon_value:
            geo_data_dir = f"geo_data_epsilon_{epsilon_value}"
            config['dataset']['params']['geo_data_dir_candidates'] = [geo_data_dir]
            print(f"Using EPSILON_VALUE to set geo_data_dir: {geo_data_dir}", file=sys.stderr)
        else:
            # 最后使用默认值
            geo_data_dir = 'geo_data'
            config['dataset']['params']['geo_data_dir_candidates'] = [geo_data_dir]
            print(f"Using default geo_data_dir: {geo_data_dir}", file=sys.stderr)

# 写入临时配置文件
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# 输出 geo_data_dir 供 bash 使用（输出到 stdout）
print(geo_data_dir)
EOF
)

OUTPUT_DIR="/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-minimal-finetuning-dinol518_on_hunyuan_-1-1/epsilon_${EPSILON_VALUE}_mini-trainset_bs=${BS}_latents=${NUM_LATENTS}"
NUM_NODES=1
UPDATE_EVERY=2 # 梯度累积步数
CUDA_DEVICES="4,7"  # 指定使用的 GPU
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# 可选：从 checkpoint 恢复训练
# 想在不同 GPU 数量上继续训练：把脚本里的 CKPT_PATH 改成 .../ckpt-step-xxxx.fp32，运行时会自动载入模型参数并冻结不需要训练的分支。
# 想按原 ZeRO 检查点完整续训（需匹配 GPU 数量）：把 CKPT_PATH 指向 ckpt-step-xxxx.ckpt 目录即可，逻辑会自动绕过手动加载。
CKPT_PATH="${OUTPUT_DIR}/ckpt/ckpt-step=00002000-v1.ckpt/ckpt-step=00002000-v1.fp32" # 转换后的checkpoint

echo "🚀 开始第一阶段训练 (1-$BS parts)"
echo "配置文件: $CONFIG"
echo "临时配置文件: $TEMP_CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BS (从配置文件读取)"
echo "EPSILON_VALUE: $EPSILON_VALUE"
echo "使用的 geo_data 目录: $GEO_DATA_DIR_USED"
echo "GPU 数量: $NUM_GPUS"
echo "梯度累积: $UPDATE_EVERY"
echo "使用 GPU: $CUDA_DEVICES"

# 设置环境变量, 解决NCCL超时问题
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export TORCH_NCCL_BLOCKING_WAIT=1          # 或 TORCH_NCCL_BLOCKING_WAIT=1，二选一即可
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # 建议同时打开，出错能及时退出
export NCCL_TIMEOUT=1800             # 单位秒，可根据需要调大到 1800、2400 等

# CUDA 调试环境变量（定位具体错误位置，会降低性能）
export CUDA_LAUNCH_BLOCKING=1      # 同步执行 CUDA 操作，便于定位错误
export TORCH_USE_CUDA_DSA=1        # 启用设备端断言（需要重新编译 PyTorch）

# 训练命令（使用临时配置文件）
python /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/main_ours_whole.py \
    --num_nodes "$NUM_NODES" \
    --num_gpus "$NUM_GPUS" \
    --config "$TEMP_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed2 \
    --update_every "$UPDATE_EVERY" \
    # --ckpt_path "$CKPT_PATH"  # 取消注释以从 checkpoint 恢复

# 清理临时配置文件
rm -f "$TEMP_CONFIG"

echo "✅ 第一阶段训练完成"
echo "输出目录: $OUTPUT_DIR"
