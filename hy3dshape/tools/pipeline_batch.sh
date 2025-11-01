#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=5
export OPENCV_IO_ENABLE_OPENEXR=1

# 设置Blender路径（使用完整路径）
export BLENDER_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/Low-poly/LowPoly_glb #1700-2299 #1000-1699 #500-1000 #51-500 #0-50
OUT_ROOT=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-lowpoly/rendered_imgs_by_hunyuan

# 参数
RESOLUTION=${RESOLUTION:-1024}
VIEWS=${VIEWS:-24}
ENGINE=${ENGINE:-CYCLES}
LIMIT=${LIMIT:--1}
# NAME_FILTER=${NAME_FILTER:-"_thin"}  # 文件名过滤器，只处理包含此字符串的文件

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_ROOT"

echo "[INFO] Starting single-threaded rendering..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output directory: $OUT_ROOT"
echo "[INFO] Resolution: $RESOLUTION"
echo "[INFO] Views: $VIEWS"
echo "[INFO] Engine: $ENGINE"

# 单线程处理所有文件
$BLENDER_PATH -b -P /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/render/render_batch.py -- \
  --input_dir "$INPUT_DIR" \
  --out_root "$OUT_ROOT" \
  --patterns "*.glb" \
  --resolution "$RESOLUTION" \
  --engine "$ENGINE" \
  --views "$VIEWS" \
  --limit "$LIMIT" \
  --progress_file "$OUT_ROOT/render_progress.md" \
  --geo_mode # 有的话是 shape，没有是 texture 生成需要的数据

echo "[INFO] Rendering completed!"
