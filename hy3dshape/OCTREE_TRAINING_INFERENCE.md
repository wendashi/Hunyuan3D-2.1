# Octree 与训练/推理的关系说明

## 核心结论

**octree_resolution 只在推理时使用，不影响训练过程。** 但训练过程中模型学到的 latent 表示质量会影响最终生成的 mesh 复杂度。

## 训练流程（不涉及 octree_resolution）

```
训练数据 (surface 点云)
    ↓
VAE.encode()  [冻结，不训练]
    ↓
latents (潜在表示)
    ↓
DiT 模型训练 [学习从条件生成 latents]
    ↓
损失函数优化
```

**关键点**：
1. VAE 是冻结的（`instantiate_non_trainable_model`），训练过程中不会改变
2. 训练时只处理 latents，**不生成 mesh**，因此不使用 `octree_resolution`
3. 训练数据是 surface（点云），从 `.npz` 文件中加载，格式为 `[N, 7]`（3D坐标 + 3D法线 + 1个标签）

## 推理流程（使用 octree_resolution）

```
输入图像
    ↓
DiT 模型生成
    ↓
latents (潜在表示)
    ↓
VAE.latents2mesh() 
    ├─→ volume_decoder(octree_resolution)  [生成密集网格点的 logits]
    └─→ surface_extractor(mc_level)        [Marching Cubes 提取 mesh]
    ↓
输出 mesh (vertices + faces)
```

**关键点**：
1. `octree_resolution` 控制 volume_decoder 生成的网格密度
   - `octree_resolution = 256` → 网格大小 `(257, 257, 257)` → 约 1700万 个查询点
   - `octree_resolution = 384` → 网格大小 `(385, 385, 385)` → 约 5700万 个查询点
2. 网格密度越高，Marching Cubes 能提取的 vertices/faces 越多
3. `mc_level` 控制 Marching Cubes 的 iso-surface 阈值

## 为什么训练后 vertices/faces 会减少？

### 可能原因 1：训练数据本身的复杂度（最可能）⭐⭐⭐⭐⭐

**问题**：如果训练数据的 mesh 本身被简化了（vertices/faces 较少），模型会学习到更简单的 latent 表示。

**检查方法**：
```python
import numpy as np
import glob
import os

data_dir = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan"

point_counts = []
for item_dir in glob.glob(os.path.join(data_dir, "*/geo_data/*_surface.npz")):
    try:
        data = np.load(item_dir)
        random_surface = data['random_surface']  # [N, 6] 或 [N, 7]
        point_counts.append(len(random_surface))
    except:
        pass

print(f"训练数据平均点数: {np.mean(point_counts):.0f}")
print(f"训练数据点数范围: {np.min(point_counts)} - {np.max(point_counts)}")
print(f"训练数据点数中位数: {np.median(point_counts):.0f}")
```

**如果训练数据的点数较少**（比如平均只有几万点），那么：
- VAE 编码时学到的表示就更倾向于简单形状
- 即使推理时使用高 `octree_resolution`，生成的 latent 质量也不足以支持复杂 mesh

### 可能原因 2：模型学到的 latent 表示质量下降 ⭐⭐⭐⭐

**问题**：训练过程中，DiT 模型可能学会了用更简单的 latent 表示来拟合训练数据，导致生成的 mesh 细节减少。

**可能原因**：
- 学习率过高，导致模型过度简化
- 训练数据质量不一致
- 损失函数权重问题

**检查方法**：
```python
# 对比训练前后的 latent 分布
import torch

# 加载训练前和训练后的 checkpoint
ckpt_before = torch.load("gs-0000000000.ckpt")
ckpt_after = torch.load("gs-0000012500.ckpt")

# 检查 latent 的统计信息
# （需要实际运行推理获取 latent）
```

### 可能原因 3：volume_decoder 类型不同 ⭐⭐⭐

**问题**：不同的 volume_decoder 可能使用不同的 octree 策略。

**代码位置**：
- `VanillaVolumeDecoder`: 使用固定 `octree_resolution` 的密集网格
- `HierarchicalVolumeDecoding`: 使用多分辨率层次网格
- `FlashVDMVolumeDecoding`: 使用自适应网格

**检查方法**：
```python
# 检查 VAE 使用的 volume_decoder 类型
print(type(model.first_stage_model.volume_decoder))
```

### 可能原因 4：训练数据预处理不一致 ⭐⭐

**问题**：训练数据的预处理可能在不同阶段使用了不同的简化策略。

**检查点**：
- 数据加载时是否有简化步骤
- `pc_size` 参数是否足够大（当前配置是 81920）
- 是否有 mesh 后处理步骤

## Octree 在代码中的位置

### 1. 训练时（flow_matching_sit_ours.py）

```python
# 训练时：只编码，不涉及 octree
latents = self.first_stage_model.encode(batch[self.first_stage_key], sample_posterior=True)
# 不调用 latents2mesh，因此不使用 octree_resolution
```

### 2. 推理时（pipelines.py）

```python
# 推理时：使用 octree_resolution
outputs = self.vae.latents2mesh(
    latents,
    bounds=box_v,
    mc_level=mc_level,
    num_chunks=num_chunks,
    octree_resolution=octree_resolution,  # 这里！
    mc_algo=mc_algo,
)
```

### 3. Volume Decoder（volume_decoders.py）

```python
# VanillaVolumeDecoder 使用 octree_resolution 生成查询点
xyz_samples, grid_size, length = generate_dense_grid_points(
    bbox_min=bbox_min,
    bbox_max=bbox_max,
    octree_resolution=octree_resolution,  # 决定网格密度
    indexing="ij"
)
```

### 4. Surface Extractor（surface_extractors.py）

```python
# Marching Cubes 从网格 logits 提取 mesh
vertices, faces, normals, _ = measure.marching_cubes(
    grid_logit.cpu().numpy(),
    mc_level,  # iso-surface 阈值
    method="lewiner"
)
# vertices/faces 数量取决于 grid_logit 的分辨率（由 octree_resolution 决定）
```

## 诊断步骤

### 步骤 1：检查训练数据复杂度

```bash
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
print(f'训练数据点数统计:')
print(f'  平均: {np.mean(counts):.0f}')
print(f'  中位数: {np.median(counts):.0f}')
print(f'  最小: {np.min(counts)}')
print(f'  最大: {np.max(counts)}')
print(f'  标准差: {np.std(counts):.0f}')
"
```

### 步骤 2：对比训练前后的 latent 质量

```python
# 在 callback 中添加代码，保存训练前后的 latent 统计信息
# 检查 latent 的方差、范围等
```

### 步骤 3：检查 volume_decoder 类型

```python
# 确认使用的 volume_decoder 类型是否一致
print(f"Volume Decoder: {type(model.first_stage_model.volume_decoder)}")
print(f"Surface Extractor: {type(model.first_stage_model.surface_extractor)}")
```

### 步骤 4：对比不同 octree_resolution 的效果

```python
# 使用相同的 checkpoint，但不同的 octree_resolution 进行推理
# 看 vertices/faces 数量是否按预期变化
for resolution in [128, 256, 384, 512]:
    mesh = pipeline(..., octree_resolution=resolution)
    print(f"resolution={resolution}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
```

## 解决方案

### 1. 如果训练数据被简化了

- 使用更高分辨率的训练数据
- 增加 `pc_size` 参数（当前是 81920，可以尝试更大）
- 检查数据预处理流程，确保没有不必要的简化

### 2. 如果模型学到了简单表示

- 降低学习率
- 调整损失函数权重
- 使用更长的训练时间
- 检查是否有梯度裁剪导致的学习受限

### 3. 确保 octree_resolution 一致

- 训练和推理使用相同的 `octree_resolution`
- 确保 callback 参数正确传递

## 总结

**octree_resolution 只在推理时影响 mesh 生成，但训练数据质量和模型学到的 latent 表示质量才是决定最终 mesh 复杂度的关键因素。**

如果训练数据的 mesh 本身被简化了，或者模型学到了更简单的表示，那么即使使用相同的 `octree_resolution`，生成的 mesh 也会更简单。

**建议优先检查训练数据的复杂度。**

