#!/bin/bash

# 测试不同 epsilon 值的脚本
# 循环测试多个 epsilon 值，只输出 surface.npz

# 设置环境变量
export OPENCV_IO_ENABLE_OPENEXR=1

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/mini-trainset

# 参数
GRID_RES=${GRID_RES:-256}
LIMIT=${LIMIT:--1}

# 要测试的 epsilon 值列表（格式：分子/分母）
EPSILON_VALUES=("1.0/256" "1.5/256" "0.5/256" "0.25/256")

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

echo "[INFO] Starting epsilon testing..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Grid resolution: $GRID_RES"
echo "[INFO] Epsilon values to test: ${EPSILON_VALUES[@]}"

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
LOG_FILE="$INPUT_DIR/epsilon_test_processing.log"
PROGRESS_FILE="$INPUT_DIR/epsilon_test_progress.md"

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Progress file: $PROGRESS_FILE"

# 初始化进度文件
cat > "$PROGRESS_FILE" << EOF
# Epsilon 测试处理进度

## 处理参数
- **输入目录**: $INPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 值**: ${EPSILON_VALUES[@]}
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
    
    current=$((i + 1))
    progress=$((current * 100 / TOTAL_FILES))
    
    echo "[$current/$TOTAL_FILES] ($progress%) Processing: $dirname" | tee -a "$LOG_FILE"
    
    # 对每个 epsilon 值进行处理
    for epsilon_expr in "${EPSILON_VALUES[@]}"; do
        # 计算 epsilon 的数值
        epsilon=$(echo "scale=10; $epsilon_expr" | bc)
        
        # 生成 epsilon 的标识符（用于文件名，例如 1.0/256 -> 1_0_256）
        epsilon_id=$(echo "$epsilon_expr" | tr '/' '_' | tr '.' '_')
        
        # 设置输出目录和前缀（最终文件名为 surface_${epsilon_id}_surface.npz）
        output_dir="$dir_path/geo_data_epsilon_new"
        output_prefix="$output_dir/surface_${epsilon_id}"
        
        # 创建输出目录
        mkdir -p "$output_dir"
        
        echo "  Testing epsilon=$epsilon_expr ($epsilon)..." | tee -a "$LOG_FILE"
        
        # 执行水密网格处理和采样（只输出 surface）
        /opt/liblibai-models/user-workspace/miniconda3/envs/hunyuan21_wenda/bin/python /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample_ours.py \
            --input_obj "$mesh_file" \
            --output_prefix "$output_prefix" \
            --grid_res "$GRID_RES" \
            --epsilon "$epsilon" \
            --surface_only 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed epsilon=$epsilon_expr" | tee -a "$LOG_FILE"
        else
            echo "  ✗ Failed epsilon=$epsilon_expr" | tee -a "$LOG_FILE"
            failed_count=$((failed_count + 1))
        fi
    done
    
    if [ $? -eq 0 ]; then
        success_count=$((success_count + 1))
        echo "- [$current/$TOTAL_FILES] ✅ $dirname - $(date)" >> "$PROGRESS_FILE"
    else
        failed_files+=("$dirname")
        echo "- [$current/$TOTAL_FILES] ❌ $dirname - $(date)" >> "$PROGRESS_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

# 生成最终报告
echo "[INFO] Generating final report..." | tee -a "$LOG_FILE"
final_report="$INPUT_DIR/epsilon_test_final_report.md"

cat > "$final_report" << EOF
# Epsilon 测试完成报告

## 处理参数
- **输入目录**: $INPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 值**: ${EPSILON_VALUES[@]}
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

## 输出文件命名格式
每个 epsilon 值会生成对应的 surface 文件（位于 \`geo_data_epsilon_test\` 目录下）：
- \`surface_1_0_256_surface.npz\` (epsilon=1.0/256)
- \`surface_1_5_256_surface.npz\` (epsilon=1.5/256)
- \`surface_0_5_256_surface.npz\` (epsilon=0.5/256)
- \`surface_0_25_256_surface.npz\` (epsilon=0.25/256)

## 日志文件
- 处理日志: \`epsilon_test_processing.log\`
- 进度文件: \`epsilon_test_progress.md\`

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

