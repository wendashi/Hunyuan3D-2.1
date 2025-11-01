#!/bin/bash

# 重新渲染缺失文件的脚本（修复版）
# 专门处理之前渲染失败的8个文件

# 设置环境变量
export OPENCV_IO_ENABLE_OPENEXR=1

# 设置Blender路径
export BLENDER_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/Low-poly/LowPoly_glb
OUT_ROOT=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-Img/LowPoly/cloth-only-img

# 参数
RESOLUTION=${RESOLUTION:-1024}
ENGINE=${ENGINE:-CYCLES}

# 导出环境变量供子进程使用
export RESOLUTION
export ENGINE

# 缺失的文件列表
MISSING_FILES=(
    "LowPoly_0017"
    "LowPoly_0041"
    "LowPoly_0211"
    "LowPoly_0212"
    "LowPoly_0248"
    "LowPoly_0447"
    "LowPoly_0454"
    "LowPoly_0782"
)

echo "[INFO] 开始重新渲染缺失的文件（修复版）..."
echo "[INFO] 输入目录: $INPUT_DIR"
echo "[INFO] 输出目录: $OUT_ROOT"
echo "[INFO] 分辨率: $RESOLUTION"
echo "[INFO] 渲染引擎: $ENGINE"
echo "[INFO] 缺失文件数量: ${#MISSING_FILES[@]}"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_ROOT"

# 创建日志文件
LOG_FILE="$OUT_ROOT/rerender_missing_files_fixed.log"
echo "[INFO] 日志文件: $LOG_FILE"

# 记录开始时间
start_time=$(date)
echo "[INFO] 开始时间: $start_time" | tee "$LOG_FILE"

# 处理每个缺失的文件
success_count=0
failed_count=0
failed_files=()

for i in "${!MISSING_FILES[@]}"; do
    filename="${MISSING_FILES[$i]}"
    input_file="$INPUT_DIR/${filename}.glb"
    output_file="$OUT_ROOT/${filename}.png"
    
    echo "" | tee -a "$LOG_FILE"
    echo "[$((i+1))/${#MISSING_FILES[@]}] 处理文件: $filename" | tee -a "$LOG_FILE"
    
    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        echo "[ERROR] 输入文件不存在: $input_file" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
        failed_files+=("$filename")
        continue
    fi
    
    # 检查输出文件是否已存在
    if [ -f "$output_file" ]; then
        echo "[WARN] 输出文件已存在，跳过: $output_file" | tee -a "$LOG_FILE"
        success_count=$((success_count + 1))
        continue
    fi
    
    echo "[INFO] 输入文件: $input_file" | tee -a "$LOG_FILE"
    echo "[INFO] 输出文件: $output_file" | tee -a "$LOG_FILE"
    
    # 创建临时输出目录
    temp_out_dir="$OUT_ROOT/temp_${filename}"
    mkdir -p "$temp_out_dir"
    
    # 使用Blender渲染
    echo "[INFO] 开始渲染..." | tee -a "$LOG_FILE"
    render_start_time=$(date +%s)
    
    if $BLENDER_PATH -b -P /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/render/render_batch_single_view.py -- \
        --input_dir "$INPUT_DIR" \
        --out_root "$temp_out_dir" \
        --patterns "*.glb" \
        --resolution "$RESOLUTION" \
        --engine "$ENGINE" \
        --limit 1 \
        --file_list <(echo "$input_file") \
        --geo_mode 2>&1 | tee -a "$LOG_FILE"; then
        
        render_end_time=$(date +%s)
        render_duration=$((render_end_time - render_start_time))
        echo "[INFO] 渲染完成，耗时: ${render_duration}秒" | tee -a "$LOG_FILE"
        
        # 检查渲染结果 - 修复：直接查找重命名后的文件
        rendered_file="$temp_out_dir/${filename}.png"
        if [ -f "$rendered_file" ]; then
            # 移动文件到最终位置
            if mv "$rendered_file" "$output_file"; then
                echo "[SUCCESS] 文件移动成功: $output_file" | tee -a "$LOG_FILE"
                success_count=$((success_count + 1))
            else
                echo "[ERROR] 文件移动失败: $rendered_file -> $output_file" | tee -a "$LOG_FILE"
                failed_count=$((failed_count + 1))
                failed_files+=("$filename")
            fi
        else
            echo "[ERROR] 渲染文件不存在: $rendered_file" | tee -a "$LOG_FILE"
            # 尝试查找其他可能的文件名
            echo "[DEBUG] 查找临时目录中的所有文件:" | tee -a "$LOG_FILE"
            ls -la "$temp_out_dir" | tee -a "$LOG_FILE"
            failed_count=$((failed_count + 1))
            failed_files+=("$filename")
        fi
        
        # 清理临时目录
        rm -rf "$temp_out_dir"
        
    else
        render_end_time=$(date +%s)
        render_duration=$((render_end_time - render_start_time))
        echo "[ERROR] 渲染失败，耗时: ${render_duration}秒" | tee -a "$LOG_FILE"
        failed_count=$((failed_count + 1))
        failed_files+=("$filename")
        
        # 清理临时目录
        rm -rf "$temp_out_dir"
    fi
done

# 记录结束时间
end_time=$(date)
echo "" | tee -a "$LOG_FILE"
echo "[INFO] 结束时间: $end_time" | tee -a "$LOG_FILE"
echo "[INFO] 处理完成!" | tee -a "$LOG_FILE"
echo "[INFO] 成功: $success_count 个文件" | tee -a "$LOG_FILE"
echo "[INFO] 失败: $failed_count 个文件" | tee -a "$LOG_FILE"

if [ $failed_count -gt 0 ]; then
    echo "[INFO] 失败的文件:" | tee -a "$LOG_FILE"
    for file in "${failed_files[@]}"; do
        echo "  - $file" | tee -a "$LOG_FILE"
    done
fi

# 生成最终报告
report_file="$OUT_ROOT/rerender_report_fixed.md"
cat > "$report_file" << EOF
# 重新渲染缺失文件报告（修复版）

## 渲染参数
- **输入目录**: $INPUT_DIR
- **输出目录**: $OUT_ROOT
- **分辨率**: $RESOLUTION
- **渲染引擎**: $ENGINE
- **开始时间**: $start_time
- **结束时间**: $end_time

## 处理结果
- **总文件数**: ${#MISSING_FILES[@]}
- **成功数量**: $success_count
- **失败数量**: $failed_count

## 缺失文件列表
EOF

for file in "${MISSING_FILES[@]}"; do
    if [[ " ${failed_files[*]} " =~ " $file " ]]; then
        echo "- **$file**: ❌ 失败" >> "$report_file"
    else
        echo "- **$file**: ✅ 成功" >> "$report_file"
    fi
done

cat >> "$report_file" << EOF

## 日志文件
- 详细日志: rerender_missing_files_fixed.log

---
*报告生成时间: $(date)*
EOF

echo "[INFO] 最终报告: $report_file"

if [ $failed_count -eq 0 ]; then
    echo "[SUCCESS] 所有缺失文件重新渲染成功!"
    exit 0
else
    echo "[ERROR] 有 $failed_count 个文件渲染失败，请检查日志"
    exit 1
fi