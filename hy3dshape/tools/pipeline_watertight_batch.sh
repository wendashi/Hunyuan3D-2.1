#!/bin/bash

# 水密网格处理脚本
# 基于 flexible_24view.sh 的结构，但简化为串行处理

# 设置环境变量
export OPENCV_IO_ENABLE_OPENEXR=1
# 删除无用的 BLENDER_PATH 设置

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/mini-trainset

# 参数
GRID_RES=${GRID_RES:-256}
EPSILON_VALUE=${EPSILON_VALUE:-1.5}  # EPSILON_VALUE 表示分子，实际 epsilon = EPSILON_VALUE / GRID_RES
LIMIT=${LIMIT:--1}

# 计算实际的 epsilon 值 (EPSILON_VALUE / GRID_RES)
EPSILON=$(awk "BEGIN {printf \"%.10f\", $EPSILON_VALUE / $GRID_RES}")

# 将 epsilon 分子值转换为文件夹名称（使用 EPSILON_VALUE 使路径更短）
EPSILON_DIR_NAME="$EPSILON_VALUE"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

echo "[INFO] Starting watertight processing..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Grid resolution: $GRID_RES"
echo "[INFO] Epsilon numerator: $EPSILON_VALUE"
echo "[INFO] Epsilon value: $EPSILON (=$EPSILON_VALUE/$GRID_RES)"
echo "[INFO] Output directory pattern: geo_data_epsilon_${EPSILON_DIR_NAME}"
echo "[INFO] Output mode: surface.npz only"

# 获取所有需要处理的文件
echo "[INFO] Collecting input files..."
ALL_FILES=()
while IFS= read -r subdir; do
    mesh_path="$subdir/render_cond/mesh.ply"
    if [ -f "$mesh_path" ]; then
        ALL_FILES+=("$mesh_path")
    else
        echo "[WARNING] Missing mesh.ply under: $subdir/render_cond" >&2
    fi
done < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL_FILES=${#ALL_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo "[ERROR] No mesh.ply files found in $INPUT_DIR"
    exit 1
fi

echo "[INFO] Found $TOTAL_FILES files to process"

# 如果指定了限制数量，截取文件列表
if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $TOTAL_FILES ]; then
    ALL_FILES=("${ALL_FILES[@]:0:$LIMIT}")
    TOTAL_FILES=$LIMIT
    echo "[INFO] Limited to $TOTAL_FILES files"
fi

# 创建日志文件
LOG_FILE="$INPUT_DIR/watertight_processing.log"
PROGRESS_FILE="$INPUT_DIR/watertight_progress.md"

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Progress file: $PROGRESS_FILE"

# 初始化进度文件
cat > "$PROGRESS_FILE" << EOF
# 水密网格处理进度

## 处理参数
- **输入目录**: $INPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 分子**: $EPSILON_VALUE
- **Epsilon 值**: $EPSILON (=$EPSILON_VALUE/$GRID_RES)
- **总文件数**: $TOTAL_FILES
- **开始时间**: $(date)

## 处理进度
EOF

# 开始处理
echo "[INFO] Starting processing..." | tee "$LOG_FILE"

success_count=0
failed_count=0
failed_files=()

for i in "${!ALL_FILES[@]}"; do
    mesh_file="${ALL_FILES[$i]}"
    dir_path=$(dirname $(dirname "$mesh_file"))
    dirname=$(basename "$dir_path")
    output_dir="$dir_path/geo_data_epsilon_${EPSILON_DIR_NAME}"
    output_prefix="$output_dir/$dirname"
    
    current=$((i + 1))
    progress=$((current * 100 / TOTAL_FILES))
    
    echo "[$current/$TOTAL_FILES] ($progress%) Processing: $dirname" | tee -a "$LOG_FILE"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 执行水密网格处理和采样（只输出 surface.npz）
    /opt/liblibai-models/user-workspace/miniconda3/envs/hunyuan21_wenda/bin/python /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample_ours.py \
        --input_obj "$mesh_file" \
        --output_prefix "$output_prefix" \
        --grid_res "$GRID_RES" \
        --epsilon "$EPSILON" \
        --surface_only 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "[$current/$TOTAL_FILES] ✓ Completed: $dirname" | tee -a "$LOG_FILE"
        success_count=$((success_count + 1))
        echo "- [$current/$TOTAL_FILES] ✅ $dirname - $(date)" >> "$PROGRESS_FILE"
    else
        echo "[$current/$TOTAL_FILES] ✗ Failed: $dirname" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
        failed_files+=("$dirname")
        echo "- [$current/$TOTAL_FILES] ❌ $dirname - $(date)" >> "$PROGRESS_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

# 生成最终报告
echo "[INFO] Generating final report..." | tee -a "$LOG_FILE"
final_report="$INPUT_DIR/watertight_final_report.md"

cat > "$final_report" << EOF
# 水密网格处理完成报告

## 处理参数
- **输入目录**: $INPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 分子**: $EPSILON_VALUE
- **Epsilon 值**: $EPSILON (=$EPSILON_VALUE/$GRID_RES)
- **总文件数**: $TOTAL_FILES
- **开始时间**: $(date)

## 处理结果
- **成功**: $success_count
- **失败**: $failed_count
- **成功率**: $((success_count * 100 / TOTAL_FILES))%

## 失败文件
EOF

if [ ${#failed_files[@]} -gt 0 ]; then
    for failed_file in "${failed_files[@]}"; do
        echo "- $failed_file" >> "$final_report"
    done
else
    echo "- 无" >> "$final_report"
fi

cat >> "$final_report" << EOF

## 日志文件
- 处理日志: \`watertight_processing.log\`
- 进度文件: \`watertight_progress.md\`

---
*报告生成时间: $(date)*
EOF

# 输出最终结果
echo "[INFO] Processing completed!" | tee -a "$LOG_FILE"
echo "[INFO] Success: $success_count, Failed: $failed_count" | tee -a "$LOG_FILE"
echo "[INFO] Final report: $final_report" | tee -a "$LOG_FILE"

if [ $failed_count -eq 0 ]; then
    echo "[SUCCESS] All files processed successfully!"
    exit 0
else
    echo "[WARNING] Some files failed to process. Check the log for details."
    exit 1
fi