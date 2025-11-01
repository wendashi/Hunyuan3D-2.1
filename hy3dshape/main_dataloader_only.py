# -*- coding: utf-8 -*-
"""
仅用于验证数据加载的脚本
从 main.py 中删除了模型加载、训练等部分，只保留数据加载相关代码
用于验证数据读取代码的正确性
"""

import warnings
warnings.filterwarnings("ignore")

import os
import torch
import argparse
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from hy3dshape.utils import get_config_from_file, instantiate_from_config


def merge_cfg(cfg, arg_cfg):
    """合并命令行参数到配置文件"""
    for key in arg_cfg.keys():
        if key in cfg.training:
            arg_cfg[key] = cfg.training[key]
    cfg.training = DictConfig(arg_cfg)
    return cfg


def get_args():
    """解析命令行参数（只保留必要的参数）"""
    parser = argparse.ArgumentParser(description="验证数据加载脚本")
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("-s", "--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--max_batches", type=int, default=5, help="最大验证批次数量")
    parser.add_argument("--output_dir", type=str, default="./outputs/dataloader_test", help="输出目录")
    return parser.parse_args()


def validate_batch(batch, batch_idx, batch_size):
    """
    验证批次数据格式
    
    Args:
        batch: 数据批次
        batch_idx: 批次索引
        batch_size: 期望的批次大小（parts 数量）
    """
    print(f"\n{'='*60}")
    print(f"批次 {batch_idx} 验证")
    print(f"{'='*60}")
    
    # 验证必需的键
    required_keys = ['surface', 'image', 'mask', 'num_parts']
    for key in required_keys:
        if key not in batch:
            print(f"❌ 缺少必需的键: {key}")
            return False
        print(f"✓ 存在键: {key}")
    
    # 验证数据类型和形状
    surface = batch['surface']
    image = batch['image']
    mask = batch['mask']
    num_parts = batch['num_parts']
    
    print(f"\n数据形状:")
    print(f"  surface: {surface.shape} (期望: [{batch_size}, num_points, 7])")
    print(f"  image:   {image.shape} (期望: [{batch_size}, 3, H, W])")
    print(f"  mask:    {mask.shape} (期望: [{batch_size}, 1, H, W])")
    print(f"  num_parts: {num_parts} (形状: {num_parts.shape})")
    
    # 验证批次大小
    actual_batch_size = surface.shape[0]
    if actual_batch_size != batch_size:
        print(f"❌ 批次大小不匹配: 期望 {batch_size}, 实际 {actual_batch_size}")
        return False
    
    # 验证 num_parts 总和
    num_parts_sum = num_parts.sum().item()
    if num_parts_sum != batch_size:
        print(f"❌ num_parts 总和 ({num_parts_sum}) 不等于 batch_size ({batch_size})")
        return False
    print(f"✓ num_parts 总和: {num_parts_sum} == batch_size: {batch_size}")
    
    # 验证数据类型
    if not isinstance(surface, torch.Tensor):
        print(f"❌ surface 不是 torch.Tensor")
        return False
    if not isinstance(image, torch.Tensor):
        print(f"❌ image 不是 torch.Tensor")
        return False
    if not isinstance(mask, torch.Tensor):
        print(f"❌ mask 不是 torch.Tensor")
        return False
    
    # 验证数值范围（surface 应该在合理范围内）
    surface_min, surface_max = surface.min().item(), surface.max().item()
    print(f"\n数值范围:")
    print(f"  surface: [{surface_min:.4f}, {surface_max:.4f}]")
    print(f"  image:   [{image.min().item():.4f}, {image.max().item():.4f}]")
    print(f"  mask:    [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    
    # 验证 surface 的最后一个维度（应该包含坐标、法线、可能还有标签）
    if surface.shape[-1] not in [3, 6, 7]:
        print(f"⚠️  警告: surface 的最后一个维度是 {surface.shape[-1]}，期望是 3, 6 或 7")
    
    # 详细验证 surface 点云的坐标范围
    print(f"\n{'='*60}")
    print("Surface 点云坐标验证")
    print(f"{'='*60}")
    
    # 提取坐标部分（前3个维度）
    coords = surface[:, :, :3]  # [batch_size, num_points, 3]
    
    # 检查每个轴的数值范围
    coord_ranges = {
        'x': (coords[:, :, 0].min().item(), coords[:, :, 0].max().item()),
        'y': (coords[:, :, 1].min().item(), coords[:, :, 1].max().item()),
        'z': (coords[:, :, 2].min().item(), coords[:, :, 2].max().item()),
    }
    
    tolerance = 0.01  # 允许的误差范围
    range_check_passed = True
    
    print(f"\n坐标范围检查 (期望: [0, 1], 容差: ±{tolerance}):")
    for axis in ['x', 'y', 'z']:
        axis_min, axis_max = coord_ranges[axis]
        if axis_min < -tolerance or axis_max > 1.0 + tolerance:
            print(f"  ❌ {axis}: [{axis_min:.6f}, {axis_max:.6f}] - 超出范围！")
            range_check_passed = False
        elif axis_min < 0 or axis_max > 1.0:
            print(f"  ⚠️  {axis}: [{axis_min:.6f}, {axis_max:.6f}] - 轻微超出，但在容差范围内")
        else:
            print(f"  ✓  {axis}: [{axis_min:.6f}, {axis_max:.6f}] - 在范围内")
    
    # 计算每个 part 的点云质心（中心点）
    centroid_tolerance = 0.1  # 质心检查的容差可以稍大一些，因为物体可能不是完全居中的
    print(f"\n点云质心检查 (期望: ~(0.5, 0.5, 0.5), 容差: ±{centroid_tolerance}):")
    centroid_check_passed = True
    
    for part_idx in range(batch_size):
        part_coords = coords[part_idx]  # [num_points, 3]
        centroid = part_coords.mean(dim=0)  # [3]
        centroid_x, centroid_y, centroid_z = centroid[0].item(), centroid[1].item(), centroid[2].item()
        
        expected_centroid = torch.tensor([0.5, 0.5, 0.5])
        centroid_diff = torch.abs(centroid - expected_centroid)
        max_diff = centroid_diff.max().item()
        
        if max_diff > centroid_tolerance:
            print(f"  ❌ Part {part_idx}: 质心 ({centroid_x:.6f}, {centroid_y:.6f}, {centroid_z:.6f}) - "
                  f"与期望值 (0.5, 0.5, 0.5) 差异过大 (最大差异: {max_diff:.6f})")
            centroid_check_passed = False
        else:
            print(f"  ✓  Part {part_idx}: 质心 ({centroid_x:.6f}, {centroid_y:.6f}, {centroid_z:.6f}) - "
                  f"最大差异: {max_diff:.6f}")
    
    # 综合检查结果
    if not range_check_passed or not centroid_check_passed:
        print(f"\n❌ Surface 点云验证失败")
        if not range_check_passed:
            print(f"  - 坐标范围不在 [0, 1] 范围内")
        if not centroid_check_passed:
            print(f"  - 点云质心不在 (0.5, 0.5, 0.5) 附近")
        return False
    else:
        print(f"\n✓  Surface 点云验证通过: 坐标范围正确，质心接近期望值")
    
    print(f"\n✅ 批次 {batch_idx} 验证通过")
    return True


def main():
    args = get_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed, workers=True)
    
    # 加载配置文件
    print(f"\n{'='*60}")
    print("加载配置文件")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    config = get_config_from_file(args.config)
    
    # 合并命令行参数（简化版，只保留必要的）
    arg_dict = vars(args)
    # 只保留 training 相关的参数
    training_params = {}
    for key in ['seed']:
        if key in arg_dict:
            training_params[key] = arg_dict[key]
    
    if 'training' not in config:
        config.training = DictConfig({})
    config.training = OmegaConf.merge(config.training, DictConfig(training_params))
    
    # 打印配置信息
    rank_zero_info("\n数据集配置:")
    rank_zero_info(OmegaConf.to_yaml(config.dataset))
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据模块
    print(f"\n{'='*60}")
    print("创建数据模块")
    print(f"{'='*60}")
    try:
        data_module: pl.LightningDataModule = instantiate_from_config(config.dataset)
        print(f"✓ 数据模块创建成功: {type(data_module).__name__}")
    except Exception as e:
        print(f"❌ 数据模块创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取批次大小
    batch_size = config.dataset.params.batch_size
    print(f"批次大小 (parts 数量): {batch_size}")
    
    # 获取训练数据加载器
    print(f"\n{'='*60}")
    print("创建训练数据加载器")
    print(f"{'='*60}")
    try:
        train_dataloader = data_module.train_dataloader()
        print(f"✓ 训练数据加载器创建成功")
        print(f"数据集大小: {len(train_dataloader.dataset)}")
    except Exception as e:
        print(f"❌ 训练数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 验证数据加载
    print(f"\n{'='*60}")
    print(f"开始验证数据加载（最多 {args.max_batches} 个批次）")
    print(f"{'='*60}")
    
    success_count = 0
    total_count = 0
    
    try:
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= args.max_batches:
                break
            
            total_count += 1
            if validate_batch(batch, batch_idx, batch_size):
                success_count += 1
    except Exception as e:
        print(f"\n❌ 数据加载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 总结
    print(f"\n{'='*60}")
    print("验证总结")
    print(f"{'='*60}")
    print(f"成功批次: {success_count}/{total_count}")
    if success_count == total_count:
        print("✅ 所有批次验证通过！数据加载代码正常工作。")
    else:
        print("❌ 部分批次验证失败，请检查数据加载代码。")
    
    # 如果提供了验证数据列表，也测试验证数据加载器
    if config.dataset.params.get('val_data_list'):
        print(f"\n{'='*60}")
        print("创建验证数据加载器")
        print(f"{'='*60}")
        try:
            val_dataloader = data_module.val_dataloader()
            print(f"✓ 验证数据加载器创建成功")
            print(f"验证数据集大小: {len(val_dataloader.dataset)}")
            
            # 验证第一个批次
            val_batch = next(iter(val_dataloader))
            if validate_batch(val_batch, 0, batch_size):
                print("✅ 验证数据加载器正常工作")
            else:
                print("❌ 验证数据加载器出现问题")
        except Exception as e:
            print(f"⚠️  验证数据加载器测试失败: {e}")
    
    print(f"\n验证完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

