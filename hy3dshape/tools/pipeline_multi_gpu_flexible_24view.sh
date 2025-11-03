#!/bin/bash

# 灵活配置的多GPU并行渲染脚本
# 支持自定义GPU数量和配置

# 设置环境变量
export OPENCV_IO_ENABLE_OPENEXR=1

# 设置Blender路径
export BLENDER_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/whole-thin-version
OUT_ROOT=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/update

# 参数
export RESOLUTION=${RESOLUTION:-1024}  # 导出为环境变量，默认1024
export VIEWS=${VIEWS:-24}              # 导出为环境变量，默认24
export ENGINE=${ENGINE:-CYCLES}        # 导出为环境变量，默认CYCLES

LIMIT=${LIMIT:--1}

# GPU配置 - 可以通过环境变量自定义
if [ -n "$CUDA_DEVICES" ]; then
    # 从环境变量读取GPU列表，格式如 "0,1,2,3" 或 "0 1 2 3"
    IFS=', ' read -ra GPUS <<< "$CUDA_DEVICES"
else
    # 使用的GPU列表
    GPUS=(0,2)
fi

NUM_GPUS=${#GPUS[@]}

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_ROOT"

echo "[INFO] Starting multi-GPU rendering with $NUM_GPUS GPUs..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output directory: $OUT_ROOT"
echo "[INFO] Resolution: $RESOLUTION"
echo "[INFO] Views: $VIEWS"
echo "[INFO] Engine: $ENGINE"
echo "[INFO] GPUs: ${GPUS[*]}"

# 定义需要处理的目标名称列表（即缺失的那些名字）
TARGETS=(
    "HighPoly_0042_thin"
    "HighPoly_0043_thin"
    "HighPoly_0044_thin"
    "HighPoly_0045_thin"
    "HighPoly_0046_thin"
    "HighPoly_0047_thin"
    "HighPoly_0048_thin"
    "HighPoly_0050_thin"
    "HighPoly_0052_thin"
    "HighPoly_0055_thin"
    "HighPoly_0057_thin"
    "HighPoly_0041_thin"
    "HighPoly_1791_thin"
    "HighPoly_0608_thin"
)

# 构建find命令的过滤条件（简化版，避免数组逻辑错误）
CONDITION=$(printf -- "-name '%s.glb' -o " "${TARGETS[@]}")
CONDITION=${CONDITION% -o }  # 移除末尾多余的" -o "

# 获取所有需要处理的文件（仅匹配目标名称的.glb）
echo "[INFO] Collecting target .glb files..."
ALL_FILES=($(eval "find '$INPUT_DIR' \( $CONDITION \) | sort"))
TOTAL_FILES=${#ALL_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo "[ERROR] No target .glb files found in $INPUT_DIR"
    exit 1
fi

echo "[INFO] Found $TOTAL_FILES files to process"

# 如果指定了限制数量，截取文件列表
if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $TOTAL_FILES ]; then
    ALL_FILES=("${ALL_FILES[@]:0:$LIMIT}")
    TOTAL_FILES=$LIMIT
    echo "[INFO] Limited to $TOTAL_FILES files"
fi

# 计算每个GPU需要处理的文件数量
FILES_PER_GPU=$((TOTAL_FILES / NUM_GPUS))
REMAINDER=$((TOTAL_FILES % NUM_GPUS))

echo "[INFO] Files per GPU: $FILES_PER_GPU (remainder: $REMAINDER)"

# 创建临时目录用于存储分片信息
TEMP_DIR="/tmp/render_splits_$$"
mkdir -p "$TEMP_DIR"

# 创建分片文件
current_index=0
for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
    gpu_id=${GPUS[$gpu_index]}
    split_file="$TEMP_DIR/gpu_${gpu_id}_files.txt"
    
    # 计算当前GPU需要处理的文件数量
    if [ $gpu_index -lt $REMAINDER ]; then
        files_for_this_gpu=$((FILES_PER_GPU + 1))
    else
        files_for_this_gpu=$FILES_PER_GPU
    fi
    
    # 写入文件列表
    end_index=$((current_index + files_for_this_gpu))
    for i in $(seq $current_index $((end_index-1))); do
        if [ $i -lt $TOTAL_FILES ]; then
            echo "${ALL_FILES[$i]}" >> "$split_file"
        fi
    done
    
    echo "[INFO] GPU $gpu_id will process $files_for_this_gpu files (indices $current_index to $((end_index-1)))"
    current_index=$end_index
done

# 启动并行渲染进程
echo "[INFO] Starting parallel rendering processes..."
pids=()

for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
    gpu_id=${GPUS[$gpu_index]}
    split_file="$TEMP_DIR/gpu_${gpu_id}_files.txt"
    log_file="$OUT_ROOT/gpu_${gpu_id}_render.log"
    progress_file="$OUT_ROOT/gpu_${gpu_id}_progress.md"
    
    echo "[INFO] Starting GPU $gpu_id process..."
    
    # 在后台启动渲染进程
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "[GPU $gpu_id] Starting rendering on GPU $gpu_id" | tee "$log_file"
        echo "[GPU $gpu_id] Processing $(wc -l < "$split_file") files" | tee -a "$log_file"
        
        # 使用render_batch.py处理分片文件
        $BLENDER_PATH -b -P /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/render/render_batch.py -- \
          --input_dir "$INPUT_DIR" \
          --out_root "$OUT_ROOT" \
          --patterns "*.glb" \
          --resolution "$RESOLUTION" \
          --engine "$ENGINE" \
          --views "$VIEWS" \
          --limit -1 \
          --progress_file "$progress_file" \
          --file_list "$split_file" \
          --geo_mode 2>&1 | tee -a "$log_file"
        
        echo "[GPU $gpu_id] Completed rendering" | tee -a "$log_file"
    ) &
    
    pids+=($!)
    echo "[INFO] GPU $gpu_id process started with PID $!"
done

# 等待所有进程完成并记录退出状态
echo "[INFO] Waiting for all GPU processes to complete..."
echo "[INFO] Process PIDs: ${pids[*]}"

# 使用关联数组记录退出状态
declare -A exit_status

# 监控进度并等待进程
failed_gpus=()
completed_count=0

while [ $completed_count -lt $NUM_GPUS ]; do
    for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
        gpu_id=${GPUS[$gpu_index]}
        pid=${pids[$gpu_index]}
        
        # 如果这个进程还没有被检查过
        if [ -z "${exit_status[$gpu_id]}" ]; then
            # 非阻塞地检查进程是否完成
            if ! kill -0 "$pid" 2>/dev/null; then
                # 进程已完成，获取退出状态
                wait "$pid"
                status=$?
                exit_status[$gpu_id]=$status
                completed_count=$((completed_count + 1))
                
                if [ $status -eq 0 ]; then
                    echo "[INFO] GPU $gpu_id completed successfully"
                else
                    echo "[ERROR] GPU $gpu_id failed with exit code $status"
                    failed_gpus+=($gpu_id)
                fi
            fi
        fi
    done
    
    running_count=$((NUM_GPUS - completed_count))
    if [ $running_count -gt 0 ]; then
        echo "[INFO] $running_count GPU processes still running..."
        sleep 30
    fi
done

echo "[INFO] All processes completed."

# 清理临时文件
rm -rf "$TEMP_DIR"

# 生成最终报告
echo "[INFO] Generating final report..."
final_report="$OUT_ROOT/final_report.md"
cat > "$final_report" << EOF
# 多GPU渲染完成报告

## 渲染参数
- **输入目录**: $INPUT_DIR
- **输出目录**: $OUT_ROOT
- **分辨率**: $RESOLUTION
- **视图数**: $VIEWS
- **渲染引擎**: $ENGINE
- **总文件数**: $TOTAL_FILES
- **使用GPU数**: $NUM_GPUS
- **GPU列表**: ${GPUS[*]}

## 分片信息
- **每GPU文件数**: $FILES_PER_GPU
- **余数文件**: $REMAINDER

## 处理结果
EOF

for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
    gpu_id=${GPUS[$gpu_index]}
    status=${exit_status[$gpu_id]}
    if [[ " ${failed_gpus[*]} " =~ " $gpu_id " ]]; then
        echo "- **GPU $gpu_id**: [失败] (退出码: $status)" >> "$final_report"
    else
        echo "- **GPU $gpu_id**: [成功]" >> "$final_report"
    fi
done

cat >> "$final_report" << EOF

## 日志文件
EOF

for gpu_id in "${GPUS[@]}"; do
    echo "- GPU $gpu_id: \`gpu_${gpu_id}_render.log\`" >> "$final_report"
done

cat >> "$final_report" << EOF

## 进度文件
EOF

for gpu_id in "${GPUS[@]}"; do
    echo "- GPU $gpu_id: \`gpu_${gpu_id}_progress.md\`" >> "$final_report"
done

cat >> "$final_report" << EOF

---
*报告生成时间: $(date)*
EOF

if [ ${#failed_gpus[@]} -eq 0 ]; then
    echo "[SUCCESS] All GPU processes completed successfully!"
    echo "[INFO] Final report: $final_report"
    exit 0
else
    echo "[ERROR] Some GPU processes failed: ${failed_gpus[*]}"
    echo "[INFO] Check the log files for details"
    echo "[INFO] Final report: $final_report"
    exit 1
fi
