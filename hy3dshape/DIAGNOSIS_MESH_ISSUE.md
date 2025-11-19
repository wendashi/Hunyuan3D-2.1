# Mesh Vertices/Faces 减少问题诊断

## 问题描述
训练前：Vertices: 187940, Faces: 375930  
训练后：Vertices: 66564, Faces: 133152  
减少比例：约 64.6% 的 vertices 和 faces 丢失

## 可能原因分析

### 1. **训练数据本身的问题（最可能）** ⭐⭐⭐⭐⭐
- **症状**：如果训练数据的 mesh 本身就被简化了（vertices/faces 较少），模型会学习生成更少的细节
- **检查方法**：
  ```python
  # 检查训练数据的 mesh 复杂度
  import trimesh
  import glob
  import os
  
  data_dir = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan"
  
  vertex_counts = []
  face_counts = []
  
  for item_dir in glob.glob(os.path.join(data_dir, "*/geo_data/*_surface.npz")):
      try:
          data = np.load(item_dir)
          random_surface = data['random_surface']
          vertex_counts.append(len(random_surface))
      except:
          pass
  
  print(f"训练数据平均点数: {np.mean(vertex_counts)}")
  print(f"训练数据点数范围: {np.min(vertex_counts)} - {np.max(vertex_counts)}")
  ```

### 2. **模型学习到的 latent 表示变化** ⭐⭐⭐⭐
- **原因**：训练过程中，模型可能学会了用更少的 latent 信息表示形状，导致生成的 mesh 分辨率降低
- **影响**：即使 octree_resolution 相同，模型生成的 latent 质量下降，提取的 mesh 细节减少

### 3. **Callback 参数传递问题** ⭐⭐⭐
- **发现**：在 `flow_matching_sit_ours.py` 的 `sample` 方法中，kwargs 没有被传递给 pipeline
- **代码位置**：`hy3dshape/models/diffusion/flow_matching_sit_ours.py:388-391`
- **当前行为**：callback 传递的 `octree_depth: 8` 等参数被忽略，pipeline 使用默认值 `octree_resolution=384`
- **修复建议**：需要将 kwargs 传递给 pipeline

### 4. **octree_resolution 参数不一致** ⭐⭐
- **配置**：callback 配置 `octree_depth: 8` → `octree_resolution = 2^8 = 256`
- **默认值**：pipeline 默认 `octree_resolution=384`
- **问题**：如果训练前使用了不同的 resolution，会导致 mesh 数量差异

### 5. **后处理简化** ⭐
- **代码位置**：`hy3dshape/postprocessors.py` 中有 `FaceReducer` 类
- **默认值**：`max_facenum=40000`
- **检查**：确认 callback 或 pipeline 中是否调用了 mesh 简化

## 检查步骤

### 步骤 1：检查训练数据
```bash
# 检查训练数据中 mesh 的复杂度
python -c "
import numpy as np
import glob
import os

data_dir = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan'
counts = []
for f in glob.glob(os.path.join(data_dir, '*/geo_data/*_surface.npz')):
    try:
        d = np.load(f)
        counts.append(len(d['random_surface']))
    except:
        pass
print(f'平均点数: {np.mean(counts):.0f}, 范围: {np.min(counts)}-{np.max(counts)}')
"
```

### 步骤 2：检查 callback 参数传递
```python
# 在 flow_matching_sit_ours.py 的 sample 方法中添加调试
def sample(self, batch, output_type='trimesh', **kwargs):
    print(f"Sample kwargs: {kwargs}")  # 添加这行
    # ... 现有代码
    outputs = self.pipeline(image=image, 
                            mask=mask,
                            generator=generator,
                            **kwargs,  # 修改：传递 kwargs 而不是 additional_params
                            **additional_params)
```

### 步骤 3：检查训练前后的实际参数
```python
# 在 pipeline 的 __call__ 方法开始处添加
print(f"octree_resolution: {octree_resolution}")
print(f"kwargs: {kwargs}")
```

### 步骤 4：对比训练前后的 mesh 质量
```python
import trimesh

# 训练前
mesh_before = trimesh.load('/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-minimal-finetuning-dinol518/whole_bs=2_latents=4096/log/infer/gs-0000000000_e-000000_b-000000/cloth-only/HighPoly_00192_thin.glb')
print(f"训练前: {len(mesh_before.vertices)} vertices, {len(mesh_before.faces)} faces")

# 训练后
mesh_after = trimesh.load('/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-minimal-finetuning-dinol518/whole_bs=2_latents=4096/log/infer/gs-0000012500_e-000000_b-025000/cloth-only/HighPoly_00192_thin.glb')
print(f"训练后: {len(mesh_after.vertices)} vertices, {len(mesh_after.faces)} faces")
```

## 修复建议

### 1. 修复 kwargs 传递问题
修改 `flow_matching_sit_ours.py` 的 `sample` 方法：

```python
def sample(self, batch, output_type='trimesh', **kwargs):
    # ... 现有代码 ...
    
    # 处理 octree_depth 转换
    if 'octree_depth' in kwargs:
        kwargs['octree_resolution'] = 2 ** kwargs.pop('octree_depth')
    
    outputs = self.pipeline(image=image, 
                            mask=mask,
                            generator=generator,
                            output_type=output_type,
                            **kwargs)  # 传递所有 kwargs
```

### 2. 检查训练数据质量
确保训练数据的 mesh 有足够的细节（vertices/faces 数量）

### 3. 调整训练策略
- 如果数据确实被简化了，考虑使用更高分辨率的训练数据
- 或者调整 loss 函数，鼓励模型生成更详细的 mesh

### 4. 验证 octree_resolution 一致性
确保训练和推理时使用相同的 `octree_resolution` 参数

## 最可能的原因

根据你的描述"数据是其中一个"，**最可能的原因是训练数据本身被简化了**。如果训练数据的 mesh vertices/faces 数量较少，模型会学习生成更少的细节，导致训练后生成的 mesh 复杂度降低。

建议优先检查训练数据的复杂度，然后修复 kwargs 传递问题确保参数一致性。

