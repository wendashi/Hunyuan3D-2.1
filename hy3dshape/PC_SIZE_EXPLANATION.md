# pc_size 参数详解

## pc_size 的作用和影响

### 1. **pc_size 在数据加载时使用**（数据采样）

**位置**：`hy3dshape/data/dit_asl_ours.py:290`

```python
def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
    if self.pc_size > 0:
        # 从训练数据中采样 pc_size 个点
        ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
        random_surface = random_surface[ind]  # 从 124928 → 81920
```

**作用**：
- 将训练数据从原始点数（如 124928）采样到 `pc_size` 个点（如 81920）
- 如果原始数据点数 < `pc_size`，会使用 `replace=True` 重复采样

### 2. **pc_size 在 VAE Encoder 中使用**（模型架构）

**位置**：`hy3dshape/models/autoencoders/attention_blocks.py:597`

```python
class PointCrossAttentionEncoder:
    def __init__(self, ..., pc_size: int, ...):
        self.pc_size = pc_size  # 保存 pc_size
    
    def sample_points_and_latents(self, pc, feats):
        # 按照 pc_size 分割点云
        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size  # 确保不超过 pc_size
```

**作用**：
- Encoder 期望输入点云形状为 `[batch, pc_size + pc_sharpedge_size, dim]`
- 用于分割 random points 和 sharpedge points
- 计算 random 和 sharpedge 的 latent 数量比例

### 3. **预训练模型的 pc_size**

**发现**：预训练 VAE 模型使用 `pc_size: 81920`

```bash
# 从预训练模型配置中可以看到
cat /opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/config.yaml
# pc_size: 81920
```

## 能否修改 pc_size？

### ⚠️ **不建议直接修改 pc_size**

**原因**：

1. **预训练模型限制**：
   - 预训练的 VAE 模型是用 `pc_size=81920` 训练的
   - Encoder 内部的计算（如 latent 数量比例）基于这个值
   - 虽然输入可以小于 `pc_size`，但改变 `pc_size` 会影响内部计算

2. **Encoder 内部逻辑**：
   ```python
   # 计算 random 和 sharpedge 的 latent 数量比例
   num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
   ```
   如果修改 `pc_size`，这个比例会改变，可能影响模型性能

3. **数据流程**：
   ```
   训练数据 (124928 点)
   → 采样到 pc_size (81920 点)  ← 这里丢失信息
   → VAE Encoder 编码
   → Latents
   → DiT 训练
   ```

### ✅ **可行的解决方案**

#### 方案 1：保持 pc_size=81920，但改进采样策略（推荐）

**问题**：当前随机采样可能不均匀，丢失重要细节

**解决方案**：使用分层采样或 FPS（最远点采样）确保采样点覆盖整个形状

**代码修改**：
```python
# 在 dit_asl_ours.py 中
def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
    if self.pc_size > 0:
        if random_surface.shape[0] > self.pc_size:
            # 使用 FPS 采样而不是随机采样
            from sklearn.neighbors import NearestNeighbors
            # 或者使用其他均匀采样策略
            ind = fps_sampling(random_surface, self.pc_size)  # 需要实现 FPS
        else:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=True)
        random_surface = random_surface[ind]
```

#### 方案 2：修改 pc_size 但需要重新训练 VAE（不推荐）

**如果一定要修改 pc_size**：
1. 需要重新训练 VAE 模型
2. 需要确保新的 `pc_size` 与训练数据匹配
3. 成本很高，不推荐

#### 方案 3：调整训练超参数（最简单）

**保持 `pc_size=81920`**，但：
- 降低 `weight_decay`（从 1e-2 → 1e-3）
- 降低 `gradient_clip_val`（从 1.0 → 0.5）
- 改进采样策略（如果可能）

## 为什么训练数据点数充足但 mesh 还是简化了？

### 根本原因

虽然训练数据有 124928 个点，但：
1. **数据采样损失**：训练时被采样到 81920 点，丢失了约 34% 的信息
2. **随机采样不均匀**：可能丢失关键细节点
3. **训练超参数**：`weight_decay=1e-2` 和 `gradient_clip_val=1.0` 可能导致过度简化

### 数据流程中的信息损失

```
原始训练数据: 124928 点
    ↓ [随机采样，丢失 34%]
训练时输入: 81920 点
    ↓ [VAE Encoder]
Latents: 4096 tokens
    ↓ [DiT 训练 500 步]
生成的 Latents: 可能更简单
    ↓ [VAE Decoder + Marching Cubes]
生成的 Mesh: 81862 vertices (比训练前少 77%)
```

## 总结

1. **pc_size 是超参数**，但：
   - 预训练模型用 `pc_size=81920` 训练
   - 修改需要重新训练 VAE（成本高）
   - 不建议直接修改

2. **更好的解决方案**：
   - 保持 `pc_size=81920`
   - 改进采样策略（使用 FPS 或分层采样）
   - 调整训练超参数（降低 weight_decay 和 gradient_clip_val）

3. **信息损失的主要来源**：
   - 数据采样：124928 → 81920（丢失 34%）
   - 训练超参数导致过度简化

