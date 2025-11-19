# 参数与 Vertices/Faces 数量的直接关系

## 核心公式

```
vertices/faces 数量 = f(网格分辨率, 表面复杂度, iso-value)
```

## 参数影响表

| 参数 | 对 Vertices/Faces 的影响 | 数学关系 | 实际效果 |
|------|-------------------------|---------|---------|
| **octree_resolution** | **直接影响上限** | 网格单元数 = (octree_resolution)³ | 分辨率越高 → 可提取的 vertices/faces 越多 |
| **grid_res** | **直接影响上限** | 网格单元数 = (grid_res)³ | 分辨率越高 → 可提取的 vertices/faces 越多 |
| **grid_logits** | **决定表面复杂度** | Marching Cubes 遍历每个网格单元 | 表面越复杂 → vertices/faces 越多 |
| **mc_level** | **间接影响** | 改变提取的表面位置 | 可能改变 mesh 大小，但不直接决定数量上限 |

## 详细关系

### 1. octree_resolution → Vertices/Faces

**直接关系**：
```
grid_logits 大小 = (octree_resolution + 1)³ 个网格点
Marching Cubes 遍历 = (octree_resolution)³ 个网格单元

理论上限：
- 每个网格单元最多产生 5 个三角形（faces）
- 每个三角形有 3 个 vertices
- 最大 faces ≈ octree_resolution³ × 5
- 最大 vertices ≈ octree_resolution³ × 15（考虑共享顶点）
```

**实际例子**：
- `octree_resolution = 256` → 网格 (257, 257, 257)
  - 理论上限：~256³ × 5 ≈ 83M faces（如果表面超级复杂）
  - 实际输出：取决于 grid_logits 的内容（通常远小于上限）
  
- `octree_resolution = 128` → 网格 (129, 129, 129)
  - 理论上限：~128³ × 5 ≈ 10M faces
  - 实际输出：通常比 256 分辨率少很多

**结论**：`octree_resolution` 决定了 **vertices/faces 数量的上限**。

### 2. grid_res → Vertices/Faces（训练数据预处理）

**直接关系**：
```
预处理网格大小 = (grid_res + 1)³ 个网格点
Watertight 提取遍历 = (grid_res)³ 个网格单元

与 octree_resolution 相同：
- 最大 faces ≈ grid_res³ × 5
- 最大 vertices ≈ grid_res³ × 15
```

**实际例子**：
- `grid_res = 256` → 预处理后的 mesh 最多有 ~83M faces（理论上限）
- `grid_res = 128` → 预处理后的 mesh 最多有 ~10M faces（理论上限）

**结论**：`grid_res` 限制了 **训练数据的 vertices/faces 数量上限**。

### 3. grid_logits → Vertices/Faces

**直接关系**：
```
Marching Cubes 算法：
1. 遍历每个网格单元（共 octree_resolution³ 个）
2. 检查该单元 8 个顶点的 grid_logits 值
3. 如果穿过 iso-surface（mc_level），生成三角形
4. 生成的三角形数量 = f(表面复杂度, iso-surface 位置)
```

**关键点**：
- `grid_logits` 的值决定了表面在哪里
- 表面越复杂（更多起伏、细节）→ 更多网格单元被穿过 → 更多 faces
- 表面越简单（更平滑）→ 更少网格单元被穿过 → 更少 faces

**实际例子**：
```
情况 1：grid_logits 表示复杂表面（很多细节）
  → 很多网格单元被穿过
  → 生成很多 triangles
  → 很多 vertices/faces

情况 2：grid_logits 表示简单表面（平滑）
  → 很少网格单元被穿过
  → 生成很少 triangles
  → 很少 vertices/faces
```

**结论**：`grid_logits` 的内容决定了 **实际提取的 vertices/faces 数量**（在 octree_resolution 定义的上限内）。

### 4. mc_level → Vertices/Faces

**直接关系**：
```
mc_level 改变 iso-surface 的位置：
- mc_level = 0.0 → 提取 SDF=0 的表面（标准）
- mc_level > 0.0 → 提取更"外扩"的表面
- mc_level < 0.0 → 提取更"内缩"的表面
```

**影响**：
- `mc_level` 主要影响 **mesh 的大小和形状**，不直接决定数量上限
- 但如果表面形状变化，可能影响穿过网格单元的数量
- 通常影响较小，除非表面形状变化很大

**结论**：`mc_level` 对 vertices/faces 数量的影响 **较小**，主要影响形状。

## 实际数据示例

### 训练数据预处理（grid_res）

```python
# grid_res = 256
grid_points = (257, 257, 257)  # 约 17M 个网格点
sdf = compute_sdf(mesh, grid_points)
watertight_mesh = marching_cubes(epsilon - abs(sdf), grid_res=256)

# 实际输出：取决于原始 mesh 的复杂度
# 如果原始 mesh 简单 → 提取的 mesh 也简单
# 如果原始 mesh 复杂 → 但受 grid_res=256 限制
```

### 推理生成（octree_resolution）

```python
# octree_resolution = 256
grid_logits = vae.decode(latents, octree_resolution=256)
# grid_logits shape: (1, 257, 257, 257)

mesh = marching_cubes(grid_logits, mc_level=0.0)
# 实际输出：取决于 grid_logits 的内容
# 如果 grid_logits 简单（学习到的表示简单）→ 提取的 mesh 简单
# 如果 grid_logits 复杂 → 提取的 mesh 复杂
```

## 为什么训练后 vertices/faces 减少？

### 原因 1：grid_res 限制了训练数据 ⭐⭐⭐⭐⭐

```
训练数据预处理：
  原始 mesh (355546 vertices)
    ↓ grid_res=256 限制
  Watertight mesh (322278 vertices) ← 可能被简化了
    ↓ 训练
  模型学习简化后的表示
    ↓ 推理 octree_resolution=256
  输出 mesh (81862 vertices) ← 更简单
```

**关键**：如果 `grid_res=256` 限制了训练数据的复杂度，模型会学到更简单的表示。

### 原因 2：模型学到了更简单的 grid_logits ⭐⭐⭐⭐⭐

```
训练前（步数0）：
  grid_logits 质量好 → 表面复杂 → 355546 vertices

训练后（步数500）：
  grid_logits 质量下降 → 表面简化 → 81862 vertices
```

**关键**：训练过程中，模型可能学会了生成更简单的 `grid_logits`（更平滑的表面）。

### 原因 3：octree_resolution 的上限 ⭐⭐⭐

```
即使 grid_logits 很复杂：
  octree_resolution = 256 → 上限 ~83M faces
  octree_resolution = 128 → 上限 ~10M faces
```

**关键**：如果 `octree_resolution` 太小，即使 `grid_logits` 复杂，也无法提取更多 vertices/faces。

## 总结公式

```
最终 vertices/faces 数量 = min(
    理论上限(octree_resolution),
    实际复杂度(grid_logits)
)
```

**其中**：
- `理论上限(octree_resolution)` = f(octree_resolution³)
- `实际复杂度(grid_logits)` = Marching Cubes 遍历 grid_logits 时穿过的网格单元数

**关键理解**：
1. **octree_resolution** 和 **grid_res** 决定了 **上限**
2. **grid_logits** 的内容决定了 **实际值**（在上限内）
3. **mc_level** 影响较小，主要改变形状

## 诊断建议

如果 vertices/faces 减少了，检查：

1. **octree_resolution 是否足够大？**
   ```python
   # 如果输出 81862 vertices，但 octree_resolution=256
   # 理论上限是 ~83M faces，所以不是上限问题
   # → 问题在 grid_logits 的内容
   ```

2. **grid_res 是否限制了训练数据？**
   ```python
   # 检查训练数据预处理后的 mesh
   # 如果预处理后 vertices/faces 减少了，就是 grid_res 的问题
   ```

3. **grid_logits 的质量是否下降了？**
   ```python
   # 检查训练前后的 grid_logits
   # 如果训练后 grid_logits 更平滑（方差更小），就是模型学习的问题
   ```

