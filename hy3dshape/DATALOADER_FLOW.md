# 数据加载流程文档

本文档整理从 `train_stage1.sh` → `main.py` → `dit_asl_ours.py` 的整个数据加载流程。

## 流程概览

```
train_stage1.sh
    ↓
main.py (加载配置，创建数据模块)
    ↓
dit_asl_ours.py (MultiPartAlignedShapeLatentModule)
    ↓
MultiPartAlignedShapeLatentDataset
    ↓
collate_multi_part_batch (批次打包)
    ↓
DataLoader → Batch Data
```

## 详细流程

### 1. 训练脚本 (`train_stage1.sh`)

```bash
python main.py \
    --config configs/ours_multipart_stage1.yaml \
    --output_dir /path/to/output
```

**作用**: 调用 `main.py`，传入配置文件路径和输出目录。

---

### 2. 主程序 (`main.py`)

#### 2.1 配置加载

```python
# 加载 YAML 配置文件
config = get_config_from_file(args.config)  # 使用 OmegaConf 加载
config = merge_cfg(config, vars(args))        # 合并命令行参数
```

**配置文件结构** (`configs/ours_multipart_stage1.yaml`):
```yaml
dataset:
  target: hy3dshape.data.dit_asl_ours.MultiPartAlignedShapeLatentModule
  params:
    batch_size: 8  # 每个批次包含的 parts 数量
    train_data_list: /path/to/train.json
    val_data_list: /path/to/val.json
    pc_size: 81920
    pc_sharpedge_size: 0
    sharpedge_label: true
    return_normal: true
    image_size: 518
    # ... 其他参数
```

#### 2.2 数据模块创建

```python
# 使用 instantiate_from_config 从配置创建数据模块
data: pl.LightningDataModule = instantiate_from_config(config.dataset)
```

**工作原理**:
- `instantiate_from_config` 从 `config.dataset.target` 获取类路径
- 导入类: `hy3dshape.data.dit_asl_ours.MultiPartAlignedShapeLatentModule`
- 使用 `config.dataset.params` 作为参数实例化

---

### 3. 数据模块 (`dit_asl_ours.py`)

#### 3.1 MultiPartAlignedShapeLatentModule

继承自 `pytorch_lightning.LightningDataModule`，负责创建数据加载器。

**关键方法**:

```python
def train_dataloader(self):
    """创建训练数据加载器"""
    dataset = MultiPartAlignedShapeLatentDataset(
        data_list=self.train_data_list,
        image_transform=self.train_image_transform,
        pc_size=self.pc_size,
        batch_size=self.batch_size,  # 用于预批次打包
        # ... 其他参数
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # DataLoader 的 batch_size=1，因为已经预打包
        collate_fn=lambda batch: collate_multi_part_batch(batch[0], self.batch_size)
    )
```

#### 3.2 MultiPartAlignedShapeLatentDataset

继承自 `torch.utils.data.Dataset`，负责加载单个样本。

**初始化流程**:

1. **解析数据列表**:
   - 支持 JSON 格式: `[{"surface_path": "...", "hunyuan_images_path": "...", "num_parts": 3}, ...]`
   - 支持目录路径: `["/path/to/obj1", "/path/to/obj2", ...]`

2. **预批次打包** (如果提供 `batch_size`):
   ```python
   # 过滤: 只保留 num_parts <= batch_size 的数据
   self.data_items = [item for item in self.data_items if item['num_parts'] <= batch_size]
   # 预打包成批次，确保每个批次的总 parts 数等于 batch_size
   self.batched_items = self._get_batched_items(self.data_items, self.batch_size)
   ```

3. **`__getitem__` 方法**:
   ```python
   def __getitem__(self, idx):
       if self.batched_items is not None:
           # 预批次模式: 返回一个批次的所有物体
           batch_items = self.batched_items[idx]
           batch_parts_data = []
           for item in batch_items:
               parts_data = self._load_all_parts(item)  # 加载该物体的所有 parts
               if parts_data:
                   batch_parts_data.append(parts_data)
           return batch_parts_data
   ```

4. **`_load_all_parts` 方法**:
   - 从 `.npy` 文件或 `.npz` 文件加载所有 parts
   - 对每个 part 进行坐标转换 (Z-up → Y-up)
   - 返回所有 parts 的数据列表

#### 3.3 坐标转换

**输入格式** (Z-up 坐标系):
- `points`: `[P, 3]`, `x ∈ [-1, 1]`, `y ∈ [-1, 1]`, `z ∈ [0, 2]`
- `normals`: `[P, 3]`, Z-up 坐标系

**转换流程**:
```python
# 1. 旋转: Z-up → Y-up
points_y_up = rotate_z_up_to_y_up(points)  # (x, y, z) → (x, z, -y)

# 2. 缩放: 各轴缩放到 [0, 1]
points_final = scale_to_target_ranges(points_y_up)
# x: [-1, 1] → [0, 1] via (x + 1) / 2
# y: [0, 2] → [0, 1] via y / 2
# z: [-1, 1] → [0, 1] via (z + 1) / 2

# 3. 法线旋转 (不缩放)
normals_y_up = transform_normals_z_up_to_y_up(normals)
```

**输出格式** (Y-up 坐标系):
- `points`: `[P, 3]`, `x, y, z ∈ [0, 1]`
- `normals`: `[P, 3]`, Y-up 坐标系，已归一化

#### 3.4 批次打包 (collate_multi_part_batch)

**输入**: `batch_list = [物体1的parts列表, 物体2的parts列表, ...]`

**处理流程**:
```python
def collate_multi_part_batch(batch_list, batch_size):
    # 1. 过滤空占位符
    batch_list = [obj_parts for obj_parts in batch_list if len(obj_parts) > 0]
    
    # 2. 展平所有 parts
    all_parts = []
    num_parts_per_object = []
    for obj_parts in batch_list:
        all_parts.extend(obj_parts)
        num_parts_per_object.append(len(obj_parts))
    
    # 3. 验证 parts 总数必须等于 batch_size
    assert len(all_parts) == batch_size
    
    # 4. 打包成 batch
    surfaces = torch.stack([p['surface'] for p in all_parts])
    images = torch.stack([p['image'] for p in all_parts])
    masks = torch.stack([p['mask'] for p in all_parts])
    
    return {
        'surface': surfaces,    # [batch_size, num_points, 7]
        'image': images,        # [batch_size, 3, H, W]
        'mask': masks,          # [batch_size, 1, H, W]
        'num_parts': torch.tensor(num_parts_per_object)  # [M], 总和 = batch_size
    }
```

**输出格式**:
- `surface`: `[batch_size, pc_size, 7]`
  - 维度说明: `[x, y, z, nx, ny, nz, label]`
  - `label`: 如果是 sharpedge_label=True，最后一维是标签 (0 或 1)
- `image`: `[batch_size, 3, 518, 518]`
- `mask`: `[batch_size, 1, 518, 518]`
- `num_parts`: `[M]` tensor，记录每个物体有多少 parts，总和必须等于 `batch_size`

---

## 数据格式说明

### 输入数据格式

#### 1. JSON 格式 (PartCrafter 格式)
```json
[
  {
    "file": "obj_001",
    "surface_path": "/path/to/obj_001.npy",
    "hunyuan_images_path": "/path/to/images",
    "num_parts": 3,
    "valid": true
  },
  ...
]
```

`.npy` 文件结构:
```python
data = np.load('obj_001.npy', allow_pickle=True).item()
# data['parts'] = [
#   {
#     'surface_points': np.array([P, 3]),  # Z-up 坐标系
#     'surface_normals': np.array([P, 3])  # Z-up 坐标系
#   },
#   ...
# ]
```

#### 2. 目录路径格式 (原始 Hunyuan3D 格式)
```
/path/to/obj_001/
  ├── geo_data/
  │   ├── obj_001_part0_surface.npz
  │   ├── obj_001_part1_surface.npz
  │   └── obj_001_part2_surface.npz
  ├── render_cond/
  │   ├── 000.png
  │   ├── 001.png
  │   └── ...
  └── metadata.json  # {"num_parts": 3}
```

`.npz` 文件结构:
```python
data = np.load('obj_001_part0_surface.npz')
# data['random_surface'] = np.array([P, 6])  # [x, y, z, nx, ny, nz], 已经是 Y-up 坐标系
# data['sharp_surface'] = np.array([S, 6])    # 锐边点云（可选）
```

---

## 验证脚本

使用 `main_dataloader_only.py` 来验证数据加载流程:

```bash
python main_dataloader_only.py \
    --config configs/ours_multipart_stage1.yaml \
    --max_batches 5 \
    --output_dir ./outputs/dataloader_test
```

**脚本功能**:
- ✅ 加载配置文件
- ✅ 创建数据模块
- ✅ 创建数据加载器
- ✅ 迭代几个批次验证数据格式
- ✅ 检查数据形状、类型、数值范围
- ✅ 验证 `num_parts` 总和是否等于 `batch_size`

**输出示例**:
```
============================================================
批次 0 验证
============================================================
✓ 存在键: surface
✓ 存在键: image
✓ 存在键: mask
✓ 存在键: num_parts

数据形状:
  surface: torch.Size([8, 81920, 7]) (期望: [8, num_points, 7])
  image:   torch.Size([8, 3, 518, 518]) (期望: [8, 3, H, W])
  mask:    torch.Size([8, 1, 518, 518]) (期望: [8, 1, H, W])
  num_parts: tensor([2, 3, 3]) (形状: torch.Size([3]))
✓ num_parts 总和: 8 == batch_size: 8

✅ 批次 0 验证通过
```

---

## 常见问题

### Q1: 为什么 DataLoader 的 batch_size=1？
**A**: 因为数据集已经预打包了批次。每个样本返回一个批次的所有物体，DataLoader 只需要包装一层即可。

### Q2: num_parts 总和必须等于 batch_size 吗？
**A**: 是的。这是 PartCrafter 的逻辑，确保每个批次正好有 `batch_size` 个 parts。

### Q3: 如果某个物体的 parts 数量超过 batch_size 怎么办？
**A**: 在初始化时会过滤掉这些数据:
```python
self.data_items = [item for item in self.data_items if item['num_parts'] <= batch_size]
```

### Q4: 坐标转换是必须的吗？
**A**: 对于 PartCrafter 格式的 `.npy` 文件，是必须的，因为数据是 Z-up 坐标系。对于原始 Hunyuan3D 格式的 `.npz` 文件，数据已经是 Y-up 坐标系，不需要转换。

---

## 总结

整个数据加载流程的关键点:

1. **配置文件驱动**: 使用 YAML 配置文件定义数据模块及其参数
2. **预批次打包**: 在数据集级别进行批次打包，确保每个批次的总 parts 数等于 batch_size
3. **坐标转换**: 将 PartCrafter 格式的 Z-up 坐标系转换为 Hunyuan3D 需要的 Y-up 坐标系
4. **多格式支持**: 支持 JSON 格式和目录路径格式两种输入
5. **严格的批次验证**: 确保每个批次的数据格式和数量正确

