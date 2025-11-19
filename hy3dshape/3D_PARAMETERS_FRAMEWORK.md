# 3D 网格生成参数框架说明

## 核心概念关系图

```
训练数据预处理阶段：
    原始 Mesh → [grid_res网格] → Watertight Mesh → [pc_size采样] → 训练点云
                                                          ↓
训练阶段：
    训练点云 → VAE.encode() → latents → DiT训练
                                                          ↓
推理阶段：
    latents → VAE.decode() → grid_logits[octree_resolution] → [mc_level提取] → 最终Mesh
```

## 参数对比表

| 参数 | 使用阶段 | 作用 | 影响 | 典型值 |
|------|---------|------|------|--------|
| **grid_res** | 训练数据预处理 | 定义 watertight 处理的网格分辨率 | 限制训练数据的复杂度上限 | 256 |
| **octree_resolution** | 推理 | 定义 VAE 解码输出的网格分辨率（grid_logits 大小） | 决定最终 mesh 的细节程度 | 256, 384 |
| **grid_logits** | 推理 | VAE 解码器输出的 3D 网格张量（每个点有 SDF/logits 值） | 是提取 mesh 的"原材料" | shape: (batch, octree_resolution+1, octree_resolution+1, octree_resolution+1) |
| **mc_level** | 推理 | Marching Cubes 的 iso-value 阈值 | 控制从 grid_logits 提取表面的位置 | 0.0 |

## 详细说明

### 1. grid_res (训练数据预处理)
- **位置**: `watertight_and_sample.py`
- **作用**: 定义预处理时计算 SDF 和提取 mesh 的网格大小
- **公式**: 网格大小 = (grid_res+1, grid_res+1, grid_res+1)
- **影响**: 
  - 值越大 → 训练数据越复杂
  - 值越小 → 训练数据被简化，高频细节丢失

### 2. octree_resolution (推理)
- **位置**: VAE 解码器和 surface extractor
- **作用**: 定义推理时生成 grid_logits 的分辨率
- **公式**: grid_logits 大小 = (octree_resolution+1, octree_resolution+1, octree_resolution+1)
- **影响**: 
  - 值越大 → 最终 mesh 越详细（vertices/faces 越多）
  - 值越小 → 最终 mesh 越简单

### 3. grid_logits (推理)
- **类型**: torch.Tensor, shape = (batch, H, W, D)
- **内容**: 每个网格点的 SDF 值或 logits
- **来源**: `VAE.volume_decoder(latents, octree_resolution=...)`
- **用途**: 作为 Marching Cubes 的输入，提取 mesh 表面

### 4. mc_level (推理)
- **位置**: `MCSurfaceExtractor.run()`
- **作用**: Marching Cubes 算法的 iso-value，决定从哪个等值面提取 mesh
- **影响**: 
  - 值越大 → 提取的表面越"外扩"（mesh 变大）
  - 值越小 → 提取的表面越"内缩"（mesh 变小）
  - 通常设为 0.0（提取 SDF=0 的表面）

## 关键关系

### grid_res vs octree_resolution
- **训练数据预处理**使用 `grid_res`（限制训练数据复杂度）
- **推理**使用 `octree_resolution`（决定输出 mesh 复杂度）
- **两者独立**：训练数据可以用 `grid_res=256`，推理可以用 `octree_resolution=384`

### grid_logits 与 octree_resolution
- `grid_logits` 的分辨率由 `octree_resolution` 决定
- `octree_resolution` 越大 → `grid_logits` 越详细 → 提取的 mesh 越复杂

### mc_level 与 grid_logits
- `grid_logits` 提供 3D 场的数值
- `mc_level` 决定从哪个等值面提取表面
- 两者结合通过 Marching Cubes 算法生成 mesh

## 实际流程示例

### 训练数据预处理
```python
# watertight_and_sample.py
grid_res = 256  # 网格分辨率
grid_points = 创建网格(grid_res)  # (257, 257, 257)
sdf = 计算SDF(原始mesh, grid_points)
mesh = igl.marching_cubes(epsilon - abs(sdf), grid_res)  # 提取watertight mesh
```

### 推理生成
```python
# 1. VAE解码
octree_resolution = 256
grid_logits = vae.volume_decoder(latents, octree_resolution=octree_resolution)
# grid_logits shape: (1, 257, 257, 257)

# 2. 表面提取
mc_level = 0.0
vertices, faces = marching_cubes(grid_logits, mc_level)
```

## 总结

- **grid_res**: 训练数据预处理的"分辨率上限"
- **octree_resolution**: 推理输出的"分辨率上限"
- **grid_logits**: 推理时的"中间产物"（3D网格数值）
- **mc_level**: 从 grid_logits 提取表面的"阈值"

**核心原理**: 训练数据用 `grid_res` 限制复杂度，推理时用 `octree_resolution` 控制输出复杂度，两者可以不同。

