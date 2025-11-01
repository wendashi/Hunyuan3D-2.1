#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=7
export OPENCV_IO_ENABLE_OPENEXR=1

# 设置Blender路径（使用完整路径）
export BLENDER_PATH=/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender

# # 设置conda环境路径
# export CONDA_ENV_PATH=/opt/liblibai-models/user-workspace/miniconda3/envs/blender_wenda

# # 激活conda环境
# source /opt/liblibai-models/user-workspace/miniconda3/etc/profile.d/conda.sh
# conda activate $CONDA_ENV_PATH

# 数据路径设置
export INPUT_FILE=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data/DiFa/glbs/MD0-51-glb/00002.glb
export OUTPUT_FOLDER=/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data/DiFa/prepross-test
export NAME=00002

# 执行渲染
$BLENDER_PATH -b -P /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools/render/render.py -- \
    --object ${INPUT_FILE} \
    --output_folder $OUTPUT_FOLDER/$NAME/render_cond \
    --geo_mode \
    --resolution 4096 

# # 执行水密网格处理和采样
# python3 watertight/watertight_and_sample.py \
#     --input_obj $OUTPUT_FOLDER/$NAME/render_cond/mesh.ply \
#     --output_prefix $OUTPUT_FOLDER/$NAME/geo_data/$NAME
