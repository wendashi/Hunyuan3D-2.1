# 单视图渲染工具说明

## 概述

本工具提供了单视图渲染功能，专门用于快速渲染3D模型的正视图。相比多视图渲染，单视图渲染速度更快，适合快速预览和批量处理。

## 文件说明

### 1. render_batch_single_view.py
- **功能**: 批量单视图渲染脚本
- **特点**: 每个模型只渲染一个正视图（000.png）
- **用法**: 通过Blender调用，支持批量处理多个模型文件

### 2. pipeline_multi_gpu_flexible_1view.sh
- **功能**: 多GPU并行单视图渲染pipeline
- **特点**: 支持多GPU并行处理，自动分片文件
- **用法**: 直接运行bash脚本

### 3. render.py (已修改)
- **新增功能**: 支持单视图渲染模式
- **新增函数**: `single_front_view_camera_sequence()` - 生成正视图相机序列

## 使用方法

### 方法1: 使用多GPU Pipeline脚本（推荐）

```bash
# 激活环境
export PATH="/opt/liblibai-models/user-workspace/miniconda3/envs/hunyuan21_wenda/bin:$PATH"
export CONDA_DEFAULT_ENV="hunyuan21_wenda"
export CONDA_PREFIX="/opt/liblibai-models/user-workspace/miniconda3/envs/hunyuan21_wenda"

# 运行单视图渲染
cd /opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/tools
./pipeline_multi_gpu_flexible_1view.sh
```

### 方法2: 直接使用Python脚本

```bash
# 通过Blender调用
/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/blender/blender -b -P render_batch_single_view.py -- \
  --input_dir /path/to/input \
  --out_root /path/to/output \
  --resolution 1024 \
  --engine CYCLES
```

## 参数说明

### render_batch_single_view.py 参数
- `--input_dir`: 输入模型文件目录
- `--out_root`: 输出根目录
- `--patterns`: 文件匹配模式（默认: *.glb, *.gltf, *.obj）
- `--resolution`: 渲染分辨率（默认: 512）
- `--engine`: 渲染引擎（默认: CYCLES）
- `--geo_mode`: 几何模式渲染
- `--limit`: 处理文件数量限制（-1表示全部）
- `--name_filter`: 文件名过滤
- `--progress_file`: 进度文件路径
- `--file_list`: 指定文件列表

### pipeline_multi_gpu_flexible_1view.sh 环境变量
- `CUDA_DEVICES`: 指定使用的GPU（如: "0,1,2,3"）
- `RESOLUTION`: 渲染分辨率（默认: 1024）
- `ENGINE`: 渲染引擎（默认: CYCLES）
- `LIMIT`: 处理文件数量限制

## 输出说明

### 文件结构
```
output_directory/
├── model_name.png           # 正视图渲染结果
├── model_name_albedo.png    # 反照率图
├── model_name_mr.png        # 金属粗糙度图
├── model_name_normal.jpg    # 法线图
├── model_name_pos.jpg       # 位置图
├── model_name_transforms.json  # 相机参数
├── gpu_0_render.log         # GPU 0 渲染日志
├── gpu_0_progress.md        # GPU 0 进度文件
└── final_report_single_view.md  # 最终报告
```

### 渲染特点
- **单视图**: 每个模型只渲染一个正视图
- **直接输出**: 文件直接保存到输出目录，文件名与输入模型相同
- **快速**: 相比24视图渲染，速度提升约24倍
- **完整**: 包含所有渲染通道（albedo, normal, depth等）
- **并行**: 支持多GPU并行处理

## 注意事项

1. **环境要求**: 需要激活 `hunyuan21_wenda` conda环境
2. **GPU配置**: 默认使用GPU 0, 5, 7，可通过环境变量自定义
3. **文件格式**: 支持 .glb, .gltf, .obj 等格式
4. **输出路径**: 确保输出目录有足够的存储空间
5. **Blender路径**: 确保Blender路径正确配置

## 性能对比

| 渲染模式 | 视图数 | 相对速度 | 适用场景 |
|---------|--------|----------|----------|
| 多视图渲染 | 24 | 1x | 完整训练数据 |
| 单视图渲染 | 1 | ~24x | 快速预览、测试 |

## 故障排除

1. **环境问题**: 确保conda环境正确激活
2. **GPU问题**: 检查GPU可用性和CUDA配置
3. **路径问题**: 验证输入输出路径存在且可写
4. **权限问题**: 确保脚本有执行权限
5. **Blender问题**: 检查Blender安装和路径配置
