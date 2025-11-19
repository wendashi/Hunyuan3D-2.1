#!/bin/bash

# 部件级水密与采样批处理脚本
# 使用 watertight_and_sample_ours_part.py，对每个 mesh 按部件拆分并汇总输出

# 环境变量
export OPENCV_IO_ENABLE_OPENEXR=1

# 基础参数
GRID_RES=${GRID_RES:-256}
EPSILON_NUMERATOR=${EPSILON_NUMERATOR:-2}  # 实际 epsilon = EPSILON_NUMERATOR / GRID_RES
LIMIT=${LIMIT:--1}

# 输入/输出目录
INPUT_DIR=${INPUT_DIR:-/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train_Ours/num-part=2/raw-data}
OUTPUT_DIR=${OUTPUT_DIR:-/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train_Ours/num-part=2/part-data_epsilon_${EPSILON_NUMERATOR}}

# 部件过滤参数
MIN_FACES=${MIN_FACES:-5}
MIN_VOLUME=${MIN_VOLUME:-0.000007}
MAX_PARTS=${MAX_PARTS:-30}
SKIP_TRIM_FILTER=${SKIP_TRIM_FILTER:-0}
TRIM_KEYWORDS=${TRIM_KEYWORDS:-"trim bindedtrim topstitch edging delete me"}

# 采样参数
SAMPLE_NUM=${SAMPLE_NUM:-124928}  # 每个部件和整体的采样点数

# 并行处理参数
# 默认使用 16 个并行任务，可根据实际情况调整
# 注意：此脚本只使用 CPU，不涉及 GPU
NUM_JOBS=${NUM_JOBS:-16}  # 并行任务数，默认 16

# 数据集名称（用于 JSON 输出）
DATASET=${DATASET:-"DiFa-ours"}

# Python 解释器与脚本路径
PYTHON_BIN=${PYTHON_BIN:-/opt/liblibai-models/user-workspace/miniconda3/envs/hunyuan21_wenda/bin/python}
SCRIPT_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample_ours_part.py

# 计算实际 epsilon
EPSILON=$(awk "BEGIN {printf \"%.10f\", $EPSILON_NUMERATOR / $GRID_RES}")
EPSILON_DIR_NAME="$EPSILON_NUMERATOR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

echo "[INFO] Starting part-based watertight processing..."
echo "[INFO] Input directory: $INPUT_DIR"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Grid resolution: $GRID_RES"
echo "[INFO] Epsilon numerator: $EPSILON_NUMERATOR"
echo "[INFO] Epsilon value: $EPSILON (=$EPSILON_NUMERATOR/$GRID_RES)"
echo "[INFO] Sample number per part/object: $SAMPLE_NUM"
echo "[INFO] Part filters -> min_faces: $MIN_FACES, min_volume: $MIN_VOLUME, max_parts: $MAX_PARTS"
echo "[INFO] Skip trim filter: $([ "$SKIP_TRIM_FILTER" -eq 1 ] && echo yes || echo no)"
echo "[INFO] Trim keywords: $TRIM_KEYWORDS"
echo "[INFO] Parallel jobs: $NUM_JOBS (CPU only, total CPU cores: $(nproc))"

# 收集 GLB 文件
echo "[INFO] Collecting GLB files from input directory..."
ALL_FILES=()
while IFS= read -r glb_file; do
    if [ -f "$glb_file" ]; then
        ALL_FILES+=("$glb_file")
    fi
done < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.glb" | sort)
TOTAL_FILES=${#ALL_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    echo "[ERROR] No .glb files found in $INPUT_DIR"
    exit 1
fi

echo "[INFO] Found $TOTAL_FILES files to process"

if [ $LIMIT -gt 0 ] && [ $LIMIT -lt $TOTAL_FILES ]; then
    ALL_FILES=("${ALL_FILES[@]:0:$LIMIT}")
    TOTAL_FILES=$LIMIT
    echo "[INFO] Limited to $TOTAL_FILES files"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 日志/进度文件
LOG_FILE="$OUTPUT_DIR/watertight_part_processing.log"
PROGRESS_FILE="$OUTPUT_DIR/watertight_part_progress.md"

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Progress file: $PROGRESS_FILE"

cat > "$PROGRESS_FILE" << EOF
# 部件水密处理进度

## 处理参数
- **输入目录**: $INPUT_DIR
- **输出目录**: $OUTPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 分子**: $EPSILON_NUMERATOR
- **Epsilon 值**: $EPSILON (=$EPSILON_NUMERATOR/$GRID_RES)
- **采样点数**: $SAMPLE_NUM
- **总文件数**: $TOTAL_FILES
- **最小面数**: $MIN_FACES
- **最小体积**: $MIN_VOLUME
- **最大部件数**: $MAX_PARTS
- **跳过 Trim 过滤**: $([ "$SKIP_TRIM_FILTER" -eq 1 ] && echo 是 || echo 否)
- **Trim 关键字**: $TRIM_KEYWORDS
- **输出格式**: .npy (包含 object 和 parts 点云数据)
- **开始时间**: $(date)

## 处理进度
EOF

echo "[INFO] Starting processing..." | tee "$LOG_FILE"

success_count=0
failed_count=0
failed_files=()

# 用于收集成功处理的数据，用于生成最终 JSON
JSON_DATA_FILE="$OUTPUT_DIR/temp_json_data.txt"
# 用于并行处理的临时日志目录
TEMP_LOG_DIR="$OUTPUT_DIR/temp_logs"
mkdir -p "$TEMP_LOG_DIR"

# 用于线程安全的计数器
COUNTER_FILE="$OUTPUT_DIR/.counter"
echo "0" > "$COUNTER_FILE"

IFS=' ' read -r -a TRIM_KEYWORD_ARRAY <<< "$TRIM_KEYWORDS"

# 处理单个文件的函数
process_file() {
    local mesh_file="$1"
    local mesh_filename=$(basename "$mesh_file")
    local mesh_basename="${mesh_filename%.glb}"
    local output_file="$OUTPUT_DIR/$mesh_basename"
    local output_prefix="$output_file"
    local log_file="$TEMP_LOG_DIR/${mesh_basename}.log"
    
    # 线程安全地更新计数器（使用文件锁）
    local current
    (
        flock -n 200 || exit 1
        current=$(($(cat "$COUNTER_FILE") + 1))
        echo "$current" > "$COUNTER_FILE"
    ) 200>"$COUNTER_FILE.lock"
    
    local progress=$((current * 100 / TOTAL_FILES))
    
    echo "[$current/$TOTAL_FILES] ($progress%) Processing: $mesh_filename" > "$log_file"

    local CMD=(
        "$PYTHON_BIN"
        "$SCRIPT_PATH"
        --input_obj "$mesh_file"
        --output_prefix "$output_prefix"
        --grid_res "$GRID_RES"
        --epsilon "$EPSILON"
        --sample_num "$SAMPLE_NUM"
        --min_faces "$MIN_FACES"
        --min_volume "$MIN_VOLUME"
        --max_parts "$MAX_PARTS"
        --metadata_path "${output_prefix}_parts.json"
    )

    if [ "$SKIP_TRIM_FILTER" -eq 1 ]; then
        CMD+=(--skip_trim_filter)
    fi

    if [ ${#TRIM_KEYWORD_ARRAY[@]} -gt 0 ]; then
        CMD+=(--trim_keywords)
        for keyword in "${TRIM_KEYWORD_ARRAY[@]}"; do
            CMD+=("$keyword")
        done
    fi

    "${CMD[@]}" >> "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$current/$TOTAL_FILES] ✓ Completed: $mesh_filename" >> "$log_file"
        
        # 收集成功处理的数据用于生成 JSON
        local metadata_file="${output_prefix}_parts.json"
        local surface_path="${output_prefix}.npy"
        
        if [ -f "$metadata_file" ] && [ -f "$surface_path" ]; then
            # 从 metadata 文件读取 num_parts
            local num_parts=$($PYTHON_BIN -c "import json; data = json.load(open('$metadata_file')); print(data.get('parts_processed', 0))" 2>/dev/null || echo "0")
            
            # 线程安全地写入临时文件（使用文件锁）
            (
                flock -n 201 || exit 1
                echo "${mesh_filename}|${num_parts}|${mesh_file}|${surface_path}|${DATASET}" >> "$JSON_DATA_FILE"
            ) 201>"$JSON_DATA_FILE.lock"
        fi
        
        echo "SUCCESS|$mesh_filename" > "$TEMP_LOG_DIR/${mesh_basename}.status"
    else
        echo "[$current/$TOTAL_FILES] ✗ Failed: $mesh_filename (exit code: $exit_code)" >> "$log_file"
        echo "FAILED|$mesh_filename" > "$TEMP_LOG_DIR/${mesh_basename}.status"
    fi
}

# 导出函数和变量供子进程使用
export -f process_file
export PYTHON_BIN SCRIPT_PATH OUTPUT_DIR GRID_RES EPSILON SAMPLE_NUM
export MIN_FACES MIN_VOLUME MAX_PARTS SKIP_TRIM_FILTER
export TRIM_KEYWORD_ARRAY DATASET TOTAL_FILES COUNTER_FILE JSON_DATA_FILE TEMP_LOG_DIR

# 并行处理所有文件
echo "[INFO] Starting parallel processing with $NUM_JOBS jobs..." | tee -a "$LOG_FILE"

# 使用 xargs 进行并行处理
printf '%s\n' "${ALL_FILES[@]}" | xargs -n 1 -P "$NUM_JOBS" -I {} bash -c 'process_file "{}"'

# 汇总结果
echo "[INFO] Collecting results..." | tee -a "$LOG_FILE"
# 方法1：从状态文件读取（如果存在）
if [ -d "$TEMP_LOG_DIR" ]; then
    for status_file in "$TEMP_LOG_DIR"/*.status; do
        if [ -f "$status_file" ]; then
            read -r status mesh_filename < "$status_file"
            if [ "$status" = "SUCCESS" ]; then
                success_count=$((success_count + 1))
                echo "- ✅ $mesh_filename - $(date)" >> "$PROGRESS_FILE"
            else
                failed_count=$((failed_count + 1))
                failed_files+=("$mesh_filename")
                echo "- ❌ $mesh_filename - $(date)" >> "$PROGRESS_FILE"
            fi
        fi
    done
fi

# 方法2：如果状态文件不存在，通过检查输出文件来判断（备用方案）
if [ $success_count -eq 0 ] && [ $failed_count -eq 0 ]; then
    echo "[INFO] Status files not found, checking output files..." | tee -a "$LOG_FILE"
    for mesh_file in "${ALL_FILES[@]}"; do
        mesh_filename=$(basename "$mesh_file")
        mesh_basename="${mesh_filename%.glb}"
        output_npy="$OUTPUT_DIR/$mesh_basename.npy"
        output_json="$OUTPUT_DIR/${mesh_basename}_parts.json"
        
        if [ -f "$output_npy" ] && [ -f "$output_json" ]; then
            success_count=$((success_count + 1))
            echo "- ✅ $mesh_filename - $(date)" >> "$PROGRESS_FILE"
        else
            failed_count=$((failed_count + 1))
            failed_files+=("$mesh_filename")
            echo "- ❌ $mesh_filename - $(date)" >> "$PROGRESS_FILE"
        fi
    done
fi

# 合并所有日志文件
echo "[INFO] Merging log files..." | tee -a "$LOG_FILE"
for log_file in "$TEMP_LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        cat "$log_file" >> "$LOG_FILE"
        echo "----------------------------------------" >> "$LOG_FILE"
    fi
done

# 清理临时文件
rm -rf "$TEMP_LOG_DIR" "$COUNTER_FILE" "$COUNTER_FILE.lock" "$JSON_DATA_FILE.lock" 2>/dev/null

echo "[INFO] Generating final report..." | tee -a "$LOG_FILE"
final_report="$OUTPUT_DIR/watertight_part_final_report.md"

cat > "$final_report" << EOF
# 部件水密处理完成报告

## 处理参数
- **输入目录**: $INPUT_DIR
- **输出目录**: $OUTPUT_DIR
- **网格分辨率**: $GRID_RES
- **Epsilon 分子**: $EPSILON_NUMERATOR
- **Epsilon 值**: $EPSILON (=$EPSILON_NUMERATOR/$GRID_RES)
- **采样点数**: $SAMPLE_NUM
- **总文件数**: $TOTAL_FILES
- **最小面数**: $MIN_FACES
- **最小体积**: $MIN_VOLUME
- **最大部件数**: $MAX_PARTS
- **跳过 Trim 过滤**: $([ "$SKIP_TRIM_FILTER" -eq 1 ] && echo 是 || echo 否)
- **Trim 关键字**: $TRIM_KEYWORDS
- **输出格式**: .npy (包含 object 和 parts 点云数据)
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

## 输出文件
- 处理日志: \`watertight_part_processing.log\`
- 进度文件: \`watertight_part_progress.md\`
- 汇总 JSON: \`summary.json\` (包含所有成功处理的文件信息)

---
*报告生成时间: $(date)*
EOF

echo "[INFO] Processing completed!" | tee -a "$LOG_FILE"
echo "[INFO] Success: $success_count, Failed: $failed_count" | tee -a "$LOG_FILE"
echo "[INFO] Final report: $final_report" | tee -a "$LOG_FILE"

# 生成汇总 JSON 文件
echo "[INFO] Generating summary JSON file..." | tee -a "$LOG_FILE"
SUMMARY_JSON="$OUTPUT_DIR/summary.json"

if [ -f "$JSON_DATA_FILE" ] && [ -s "$JSON_DATA_FILE" ]; then
    $PYTHON_BIN << EOF | tee -a "$LOG_FILE"
import json
import os

json_data = []
data_file = "$JSON_DATA_FILE"

if os.path.exists(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 5:
                file_name, num_parts, mesh_path, surface_path, dataset = parts
                json_data.append({
                    "file": file_name,
                    "num_parts": int(num_parts) if num_parts.isdigit() else 0,
                    "mesh_path": mesh_path,
                    "surface_path": surface_path,
                    "dataset": dataset
                })

# 按文件名排序
json_data.sort(key=lambda x: x['file'])

# 保存 JSON 文件
with open("$SUMMARY_JSON", 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print(f"Generated summary JSON with {len(json_data)} entries: $SUMMARY_JSON")
EOF
    # 清理临时文件
    rm -f "$JSON_DATA_FILE"
else
    echo "[WARNING] No data collected for JSON summary" | tee -a "$LOG_FILE"
fi

if [ $failed_count -eq 0 ]; then
    echo "[SUCCESS] All files processed successfully!"
    exit 0
else
    echo "[WARNING] Some files failed to process. Check the log for details."
    exit 1
fi

