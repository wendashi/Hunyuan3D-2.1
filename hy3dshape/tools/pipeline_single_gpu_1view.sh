#!/bin/bash

# 基于文件列表的单视图渲染脚本
# 根据 List.txt 中的文件列表进行批量渲染，生成对应的 PNG 图像

# 设置环境变量
export OPENCV_IO_ENABLE_OPENEXR=1

# 设置Blender路径
export BLENDER_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender

# 输入/输出目录
INPUT_DIR=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/Low-poly/LowPoly_glb
OUT_ROOT=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-Img/DiFa-outfit-lowpoly/cloth-only-img-3
LIST_FILE=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/Low-poly/List_3.txt

# 参数
RESOLUTION=${RESOLUTION:-1024}
VIEWS=1  # 固定为1个视图
ENGINE=${ENGINE:-CYCLES}
LIMIT=${LIMIT:--1}

# 导出环境变量供子进程使用
export RESOLUTION
export ENGINE

# GPU配置 - 可以通过环境变量自定义
if [ -n "$CUDA_DEVICES" ]; then
    # 从环境变量读取GPU列表，格式如 "0,1,2,3" 或 "0 1 2 3"
    IFS=', ' read -ra GPUS <<< "$CUDA_DEVICES"
else
    # 默认使用3块GPU
    GPUS=(5)
fi

NUM_GPUS=${#GPUS[@]}

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_ROOT"

echo "[INFO] Starting batch single-view rendering with $NUM_GPUS GPUs..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output directory: $OUT_ROOT"
echo "[INFO] File list: $LIST_FILE"
echo "[INFO] Resolution: $RESOLUTION"
echo "[INFO] Views: $VIEWS (front view only)"
echo "[INFO] Engine: $ENGINE"
echo "[INFO] GPUs: ${GPUS[*]}"

# 获取需要处理的文件 - 基于文件列表
echo "[INFO] Collecting input files from list..."
echo "[INFO] Processing files from: $LIST_FILE"

# 检查文件列表是否存在
if [ ! -f "$LIST_FILE" ]; then
    echo "[ERROR] File list does not exist: $LIST_FILE"
    exit 1
fi

# 从文件列表中提取文件名（去掉 "  - " 前缀和 ".glb" 后缀）
echo "[INFO] Parsing file list..."
TEMP_LIST="/tmp/target_files_$$.txt"
grep -o "LowPoly_[0-9]*" "$LIST_FILE" > "$TEMP_LIST"

# 检查每个文件是否存在，并构建完整路径
ALL_FILES=()
MISSING_FILES=()

while IFS= read -r filename; do
    full_path="$INPUT_DIR/${filename}.glb"
    if [ -f "$full_path" ]; then
        ALL_FILES+=("$full_path")
    else
        MISSING_FILES+=("$filename")
    fi
done < "$TEMP_LIST"

TOTAL_FILES=${#ALL_FILES[@]}
MISSING_COUNT=${#MISSING_FILES[@]}

echo "[INFO] Found $TOTAL_FILES files to process"
if [ $MISSING_COUNT -gt 0 ]; then
    echo "[WARNING] $MISSING_COUNT files missing from input directory:"
    for file in "${MISSING_FILES[@]:0:10}"; do
        echo "  - $file"
    done
    if [ $MISSING_COUNT -gt 10 ]; then
        echo "  ... and $((MISSING_COUNT - 10)) more"
    fi
fi

if [ $TOTAL_FILES -eq 0 ]; then
    echo "[ERROR] No valid files found to process"
    rm -f "$TEMP_LIST"
    exit 1
fi

# 清理临时文件
rm -f "$TEMP_LIST"

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
TEMP_DIR="/tmp/render_splits_single_view_$$"
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
echo "[INFO] Starting parallel single-view rendering processes..."
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
        echo "[GPU $gpu_id] Starting single-view rendering on GPU $gpu_id" | tee "$log_file"
        echo "[GPU $gpu_id] Processing $(wc -l < "$split_file") files" | tee -a "$log_file"
        
        # 使用render_batch_single_view.py处理分片文件
        $BLENDER_PATH -b -P /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/render/render_batch_single_view.py -- \
          --input_dir "$INPUT_DIR" \
          --out_root "$OUT_ROOT" \
          --patterns "*.glb" \
          --resolution "$RESOLUTION" \
          --engine "$ENGINE" \
          --limit -1 \
          --progress_file "$progress_file" \
          --file_list "$split_file" \
          --geo_mode 2>&1 | tee -a "$log_file"
        
        echo "[GPU $gpu_id] Completed single-view rendering" | tee -a "$log_file"
    ) &
    
    pids+=($!)
    echo "[INFO] GPU $gpu_id process started with PID $!"
done

# 等待所有进程完成
echo "[INFO] Waiting for all GPU processes to complete..."
echo "[INFO] Process PIDs: ${pids[*]}"

# 监控进度
while true; do
    running_count=0
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running_count=$((running_count + 1))
        fi
    done
    
    if [ $running_count -eq 0 ]; then
        break
    fi
    
    echo "[INFO] $running_count GPU processes still running..."
    sleep 30
done

# 检查所有进程的退出状态
echo "[INFO] All processes completed. Checking results..."
failed_gpus=()
for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
    gpu_id=${GPUS[$gpu_index]}
    pid=${pids[$gpu_index]}
    
    if wait "$pid"; then
        echo "[INFO] GPU $gpu_id completed successfully"
    else
        echo "[ERROR] GPU $gpu_id failed"
        failed_gpus+=($gpu_id)
    fi
done

# 清理临时文件
rm -rf "$TEMP_DIR"

# 生成最终报告
echo "[INFO] Generating final report..."
final_report="$OUT_ROOT/final_report_single_view.md"
cat > "$final_report" << EOF
# 批量单视图渲染完成报告

## 渲染参数
- **输入目录**: $INPUT_DIR
- **输出目录**: $OUT_ROOT
- **文件列表**: $LIST_FILE
- **分辨率**: $RESOLUTION
- **视图数**: $VIEWS (正视图)
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
    if [[ " ${failed_gpus[*]} " =~ " $gpu_id " ]]; then
        echo "- **GPU $gpu_id**: ❌ 失败" >> "$final_report"
    else
        echo "- **GPU $gpu_id**: ✅ 成功" >> "$final_report"
    fi
done

cat >> "$final_report" << EOF

## 日志文件
EOF

for gpu_id in "${GPUS[@]}"; do
    echo "- GPU $gpu_id: gpu_${gpu_id}_render.log" >> "$final_report"
done

cat >> "$final_report" << EOF

## 进度文件
EOF

for gpu_id in "${GPUS[@]}"; do
    echo "- GPU $gpu_id: gpu_${gpu_id}_progress.md" >> "$final_report"
done

cat >> "$final_report" << EOF

## 渲染说明
- 本脚本根据 List.txt 中的文件列表进行批量渲染
- 使用单视图渲染模式，每个模型只渲染一个正视图
- 输出文件命名格式：model_name.png (与输入文件名相同)
- 渲染完成后，这些文件将可用于 PartCrafter 训练数据

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
