#!/usr/bin/env python3
"""
VAE重建质量评估脚本
用于评估预训练VAE在3D服装数据上的重建质量，判断是否需要finetune
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import json
from scipy.spatial.distance import cdist

from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.pipelines import export_to_trimesh


def compute_chamfer_distance(mesh1, mesh2, num_samples=10000):
    """
    计算两个mesh之间的Chamfer距离
    """
    # 🔥 修复：当return_index=False时，sample()只返回点云，不返回索引
    try:
        points1 = mesh1.sample(num_samples, return_index=False)
        points2 = mesh2.sample(num_samples, return_index=False)
    except Exception as e:
        # 如果return_index=False不支持，尝试return_index=True
        try:
            points1, _ = mesh1.sample(num_samples, return_index=True)
            points2, _ = mesh2.sample(num_samples, return_index=True)
        except Exception as e2:
            raise RuntimeError(f"无法从mesh采样点: {e}, {e2}")
    
    # 确保是numpy数组
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    # 检查采样点数量
    if len(points1) == 0 or len(points2) == 0:
        raise ValueError(f"采样点为空: mesh1={len(points1)}, mesh2={len(points2)}")
    
    # 计算双向最近距离
    dist_1_to_2 = cdist(points1, points2)
    dist_2_to_1 = cdist(points2, points1)
    
    chamfer_dist = (
        np.mean(np.min(dist_1_to_2, axis=1)) + 
        np.mean(np.min(dist_2_to_1, axis=1))
    ) / 2.0
    
    return chamfer_dist


def compute_hausdorff_distance(mesh1, mesh2, num_samples=10000):
    """
    计算两个mesh之间的Hausdorff距离
    """
    # 🔥 修复：当return_index=False时，sample()只返回点云，不返回索引
    try:
        points1 = mesh1.sample(num_samples, return_index=False)
        points2 = mesh2.sample(num_samples, return_index=False)
    except Exception as e:
        # 如果return_index=False不支持，尝试return_index=True
        try:
            points1, _ = mesh1.sample(num_samples, return_index=True)
            points2, _ = mesh2.sample(num_samples, return_index=True)
        except Exception as e2:
            raise RuntimeError(f"无法从mesh采样点: {e}, {e2}")
    
    # 确保是numpy数组
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    # 检查采样点数量
    if len(points1) == 0 or len(points2) == 0:
        raise ValueError(f"采样点为空: mesh1={len(points1)}, mesh2={len(points2)}")
    
    dist_1_to_2 = cdist(points1, points2)
    dist_2_to_1 = cdist(points2, points1)
    
    hausdorff_dist = max(
        np.max(np.min(dist_1_to_2, axis=1)),
        np.max(np.min(dist_2_to_1, axis=1))
    )
    
    return hausdorff_dist


def compute_volume_difference(mesh1, mesh2):
    """
    计算两个mesh的体积差异
    """
    vol1 = mesh1.volume if mesh1.is_volume else 0.0
    vol2 = mesh2.volume if mesh2.is_volume else 0.0
    
    if vol1 == 0 or vol2 == 0:
        return None
    
    vol_diff = abs(vol1 - vol2) / max(vol1, vol2)
    return vol_diff


def evaluate_vae_reconstruction(
    vae,
    surface_loader,
    mesh_path,
    device='cuda',
    dtype=torch.float16,
    octree_resolution=256,
    mc_level=0.0,
    save_reconstructed=False,
    output_dir=None,
    debug=False
):
    """
    评估单个mesh的VAE重建质量
    
    Args:
        vae: ShapeVAE模型
        surface_loader: 表面点云加载器
        mesh_path: 原始mesh路径
        device: 设备
        dtype: 数据类型
        octree_resolution: 八叉树分辨率
        mc_level: marching cubes level
        save_reconstructed: 是否保存重建的mesh
        output_dir: 输出目录
        debug: 是否输出调试信息
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    if debug:
        print(f"   🔍 [DEBUG] 开始处理: {mesh_path}")
    
    # 加载原始mesh
    try:
        if debug:
            print(f"   🔍 [DEBUG] 加载原始mesh...")
        original_mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(original_mesh, trimesh.Scene):
            if debug:
                print(f"   🔍 [DEBUG] 检测到Scene，合并mesh...")
            original_mesh = trimesh.util.concatenate(
                tuple(original_mesh.geometry.values())
            )
        
        if debug:
            print(f"   🔍 [DEBUG] 原始mesh: {len(original_mesh.vertices)} 顶点, {len(original_mesh.faces)} 面")
    except Exception as e:
        raise RuntimeError(f"加载原始mesh失败: {e}")
    
    # 加载表面点云
    try:
        if debug:
            print(f"   🔍 [DEBUG] 加载表面点云...")
        surface = surface_loader(mesh_path).to(device, dtype=dtype)
        if debug:
            print(f"   🔍 [DEBUG] 表面点云形状: {surface.shape}")
    except Exception as e:
        raise RuntimeError(f"加载表面点云失败: {e}")
    
    # VAE编码
    try:
        if debug:
            print(f"   🔍 [DEBUG] VAE编码...")
        with torch.no_grad():
            latents = vae.encode(surface)
            if debug:
                print(f"   🔍 [DEBUG] Latents形状: {latents.shape if hasattr(latents, 'shape') else type(latents)}")
            
            # VAE解码
            if debug:
                print(f"   🔍 [DEBUG] VAE解码...")
            latents = vae.decode(latents)
            if debug:
                print(f"   🔍 [DEBUG] 解码后Latents形状: {latents.shape if hasattr(latents, 'shape') else type(latents)}")
            
            # 重建mesh
            if debug:
                print(f"   🔍 [DEBUG] 重建mesh (latents2mesh)...")
            reconstructed_mesh = vae.latents2mesh(
                latents,
                output_type='trimesh',
                bounds=1.01,
                mc_level=mc_level,
                num_chunks=20000,
                octree_resolution=octree_resolution,
                mc_algo='mc',
                enable_pbar=False
            )
            if debug:
                print(f"   🔍 [DEBUG] 重建mesh完成，类型: {type(reconstructed_mesh)}")
    except Exception as e:
        raise RuntimeError(f"VAE处理失败: {e}")
    
    # 转换为trimesh格式
    try:
        if debug:
            print(f"   🔍 [DEBUG] 转换为trimesh格式...")
        reconstructed_mesh = export_to_trimesh(reconstructed_mesh)[0]
        if debug:
            print(f"   🔍 [DEBUG] 重建mesh: {len(reconstructed_mesh.vertices)} 顶点, {len(reconstructed_mesh.faces)} 面")
    except Exception as e:
        raise RuntimeError(f"转换trimesh格式失败: {e}")
    
    # 计算评估指标
    metrics = {}
    
    # 1. Chamfer距离
    try:
        if debug:
            print(f"   🔍 [DEBUG] 计算Chamfer距离...")
        chamfer_dist = compute_chamfer_distance(original_mesh, reconstructed_mesh)
        metrics['chamfer_distance'] = float(chamfer_dist)
        if debug:
            print(f"   🔍 [DEBUG] Chamfer距离: {chamfer_dist:.6f}")
    except Exception as e:
        if debug:
            print(f"   🔍 [DEBUG] Chamfer距离计算失败: {e}")
            import traceback
            traceback.print_exc()
        print(f"   ⚠️  Chamfer距离计算失败: {e}")
        metrics['chamfer_distance'] = None
    
    # 2. Hausdorff距离
    try:
        if debug:
            print(f"   🔍 [DEBUG] 计算Hausdorff距离...")
        hausdorff_dist = compute_hausdorff_distance(original_mesh, reconstructed_mesh)
        metrics['hausdorff_distance'] = float(hausdorff_dist)
        if debug:
            print(f"   🔍 [DEBUG] Hausdorff距离: {hausdorff_dist:.6f}")
    except Exception as e:
        if debug:
            print(f"   🔍 [DEBUG] Hausdorff距离计算失败: {e}")
            import traceback
            traceback.print_exc()
        print(f"   ⚠️  Hausdorff距离计算失败: {e}")
        metrics['hausdorff_distance'] = None
    
    # 3. 体积差异
    try:
        if debug:
            print(f"   🔍 [DEBUG] 计算体积差异...")
        vol_diff = compute_volume_difference(original_mesh, reconstructed_mesh)
        metrics['volume_difference'] = float(vol_diff) if vol_diff is not None else None
        if debug:
            print(f"   🔍 [DEBUG] 体积差异: {vol_diff}")
    except Exception as e:
        if debug:
            print(f"   🔍 [DEBUG] 体积差异计算失败: {e}")
            import traceback
            traceback.print_exc()
        print(f"   ⚠️  体积差异计算失败: {e}")
        metrics['volume_difference'] = None
    
    # 4. 顶点和面数统计
    metrics['original_vertices'] = len(original_mesh.vertices)
    metrics['original_faces'] = len(original_mesh.faces)
    metrics['reconstructed_vertices'] = len(reconstructed_mesh.vertices)
    metrics['reconstructed_faces'] = len(reconstructed_mesh.faces)
    
    # 5. 边界框尺寸
    try:
        orig_bbox = original_mesh.bounds
        recon_bbox = reconstructed_mesh.bounds
        orig_size = np.linalg.norm(orig_bbox[1] - orig_bbox[0])
        recon_size = np.linalg.norm(recon_bbox[1] - recon_bbox[0])
        metrics['bbox_size_ratio'] = float(recon_size / orig_size) if orig_size > 0 else None
    except Exception as e:
        if debug:
            print(f"   🔍 [DEBUG] 边界框计算失败: {e}")
        metrics['bbox_size_ratio'] = None
    
    # 保存重建的mesh
    if save_reconstructed and output_dir:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            mesh_name = Path(mesh_path).stem
            output_path = output_dir / f"{mesh_name}_reconstructed.obj"
            reconstructed_mesh.export(str(output_path))
            metrics['reconstructed_mesh_path'] = str(output_path)
            if debug:
                print(f"   🔍 [DEBUG] 保存重建mesh: {output_path}")
        except Exception as e:
            if debug:
                print(f"   🔍 [DEBUG] 保存重建mesh失败: {e}")
            print(f"   ⚠️  保存重建mesh失败: {e}")
    
    return metrics


def batch_evaluate_vae(
    vae_path_or_name='tencent/Hunyuan3D-2.1',
    mesh_dir=None,
    mesh_paths=None,
    output_dir='./vae_evaluation_results',
    num_uniform_points=81920,
    num_sharp_points=0,
    octree_resolution=256,
    mc_level=0.0,
    device='cuda',
    dtype=torch.float16,
    save_reconstructed=True,
    limit=None,
    debug=False
):
    """
    批量评估VAE重建质量
    
    Args:
        vae_path_or_name: VAE模型路径或名称
        mesh_dir: mesh目录（会查找所有.glb/.obj文件）
        mesh_paths: 指定的mesh路径列表
        output_dir: 输出目录
        num_uniform_points: 均匀采样点数
        num_sharp_points: 边缘采样点数
        octree_resolution: 八叉树分辨率
        mc_level: marching cubes level
        device: 设备
        dtype: 数据类型
        save_reconstructed: 是否保存重建mesh
        limit: 限制处理的文件数量
        debug: 是否输出调试信息
    """
    print("🚀 开始VAE重建质量评估...")
    
    # 加载VAE
    print(f"📦 加载VAE模型: {vae_path_or_name}")
    try:
        vae = ShapeVAE.from_pretrained(
            vae_path_or_name,
            use_safetensors=False,
            variant='fp16' if dtype == torch.float16 else None,
        )
        vae = vae.to(device).eval()
        print(f"✅ VAE加载成功")
    except Exception as e:
        raise RuntimeError(f"VAE加载失败: {e}")
    
    # 创建surface loader
    surface_loader = SharpEdgeSurfaceLoader(
        num_uniform_points=num_uniform_points,
        num_sharp_points=num_sharp_points,
    )
    
    # 获取mesh文件列表
    if mesh_paths:
        mesh_files = [Path(p) for p in mesh_paths]
    elif mesh_dir:
        mesh_dir = Path(mesh_dir)
        mesh_files = list(mesh_dir.rglob('*.glb')) + list(mesh_dir.rglob('*.obj'))
    else:
        raise ValueError("必须提供mesh_dir或mesh_paths")
    
    if limit:
        mesh_files = mesh_files[:limit]
    
    print(f"📋 找到 {len(mesh_files)} 个mesh文件")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_dir = output_dir / 'reconstructed_meshes'
    if save_reconstructed:
        reconstructed_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量评估
    all_metrics = []
    failed_files = []
    
    for i, mesh_path in enumerate(tqdm(mesh_files, desc="评估中")):
        print(f"\n[{i+1}/{len(mesh_files)}] 处理: {mesh_path.name}")
        try:
            metrics = evaluate_vae_reconstruction(
                vae=vae,
                surface_loader=surface_loader,
                mesh_path=str(mesh_path),
                device=device,
                dtype=dtype,
                octree_resolution=octree_resolution,
                mc_level=mc_level,
                save_reconstructed=save_reconstructed,
                output_dir=reconstructed_dir,
                debug=debug
            )
            metrics['mesh_path'] = str(mesh_path)
            metrics['mesh_name'] = mesh_path.name
            all_metrics.append(metrics)
            
            # 打印关键指标
            if metrics['chamfer_distance'] is not None:
                print(f"   ✅ Chamfer距离: {metrics['chamfer_distance']:.6f}")
            if metrics['hausdorff_distance'] is not None:
                print(f"   ✅ Hausdorff距离: {metrics['hausdorff_distance']:.6f}")
            if metrics['volume_difference'] is not None:
                print(f"   ✅ 体积差异: {metrics['volume_difference']:.4%}")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            failed_files.append({
                'mesh_path': str(mesh_path),
                'error': str(e)
            })
    
    # 计算统计信息
    print("\n📊 计算统计信息...")
    stats = {}
    
    # Chamfer距离统计
    chamfer_dists = [m['chamfer_distance'] for m in all_metrics if m['chamfer_distance'] is not None]
    if chamfer_dists:
        stats['chamfer_distance'] = {
            'mean': float(np.mean(chamfer_dists)),
            'std': float(np.std(chamfer_dists)),
            'min': float(np.min(chamfer_dists)),
            'max': float(np.max(chamfer_dists)),
            'median': float(np.median(chamfer_dists)),
        }
    
    # Hausdorff距离统计
    hausdorff_dists = [m['hausdorff_distance'] for m in all_metrics if m['hausdorff_distance'] is not None]
    if hausdorff_dists:
        stats['hausdorff_distance'] = {
            'mean': float(np.mean(hausdorff_dists)),
            'std': float(np.std(hausdorff_dists)),
            'min': float(np.min(hausdorff_dists)),
            'max': float(np.max(hausdorff_dists)),
            'median': float(np.median(hausdorff_dists)),
        }
    
    # 体积差异统计
    vol_diffs = [m['volume_difference'] for m in all_metrics if m['volume_difference'] is not None]
    if vol_diffs:
        stats['volume_difference'] = {
            'mean': float(np.mean(vol_diffs)),
            'std': float(np.std(vol_diffs)),
            'min': float(np.min(vol_diffs)),
            'max': float(np.max(vol_diffs)),
            'median': float(np.median(vol_diffs)),
        }
    
    # 保存结果
    results = {
        'config': {
            'vae_path': vae_path_or_name,
            'num_uniform_points': num_uniform_points,
            'num_sharp_points': num_sharp_points,
            'octree_resolution': octree_resolution,
            'mc_level': mc_level,
            'total_files': len(mesh_files),
            'successful': len(all_metrics),
            'failed': len(failed_files),
        },
        'statistics': stats,
        'detailed_metrics': all_metrics,
        'failed_files': failed_files,
    }
    
    # 保存JSON报告
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "="*60)
    print("📊 评估结果总结")
    print("="*60)
    print(f"总文件数: {len(mesh_files)}")
    print(f"成功: {len(all_metrics)}")
    print(f"失败: {len(failed_files)}")
    
    if stats:
        print("\n关键指标统计:")
        if 'chamfer_distance' in stats:
            s = stats['chamfer_distance']
            print(f"  Chamfer距离: 均值={s['mean']:.6f}, 中位数={s['median']:.6f}")
        if 'hausdorff_distance' in stats:
            s = stats['hausdorff_distance']
            print(f"  Hausdorff距离: 均值={s['mean']:.6f}, 中位数={s['median']:.6f}")
        if 'volume_difference' in stats:
            s = stats['volume_difference']
            print(f"  体积差异: 均值={s['mean']:.4%}, 中位数={s['median']:.4%}")
    
    print(f"\n📄 详细报告已保存: {report_path}")
    
    # 判断是否需要finetune
    print("\n" + "="*60)
    print("🔍 VAE Finetune建议")
    print("="*60)
    
    if 'chamfer_distance' in stats:
        mean_chamfer = stats['chamfer_distance']['mean']
        if mean_chamfer > 0.01:  # 阈值可调整
            print("⚠️  建议finetune VAE")
            print(f"   原因: Chamfer距离均值 ({mean_chamfer:.6f}) 较高，重建质量不佳")
        elif mean_chamfer > 0.005:
            print("💡 可考虑finetune VAE")
            print(f"   原因: Chamfer距离均值 ({mean_chamfer:.6f}) 中等，重建质量有改进空间")
        else:
            print("✅ 当前VAE表现良好，可能不需要finetune")
            print(f"   原因: Chamfer距离均值 ({mean_chamfer:.6f}) 较低，重建质量较好")
    else:
        print("⚠️  无法计算Chamfer距离，请检查数据")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估VAE重建质量')
    parser.add_argument('--vae', type=str, default='tencent/Hunyuan3D-2.1',
                        help='VAE模型路径或名称')
    parser.add_argument('--mesh_dir', type=str, default=None,
                        help='mesh文件目录')
    parser.add_argument('--mesh_paths', type=str, nargs='+', default=None,
                        help='指定的mesh文件路径列表')
    parser.add_argument('--output_dir', type=str, default='./vae_evaluation_results',
                        help='输出目录')
    parser.add_argument('--num_uniform_points', type=int, default=81920,
                        help='均匀采样点数')
    parser.add_argument('--num_sharp_points', type=int, default=0,
                        help='边缘采样点数')
    parser.add_argument('--octree_resolution', type=int, default=256,
                        help='八叉树分辨率')
    parser.add_argument('--mc_level', type=float, default=0.0,
                        help='marching cubes level')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的文件数量')
    parser.add_argument('--no_save_mesh', action='store_true',
                        help='不保存重建的mesh')
    parser.add_argument('--debug', action='store_true',
                        help='输出调试信息')
    
    args = parser.parse_args()
    
    batch_evaluate_vae(
        vae_path_or_name=args.vae,
        mesh_dir=args.mesh_dir,
        mesh_paths=args.mesh_paths,
        output_dir=args.output_dir,
        num_uniform_points=args.num_uniform_points,
        num_sharp_points=args.num_sharp_points,
        octree_resolution=args.octree_resolution,
        mc_level=args.mc_level,
        device=args.device,
        save_reconstructed=not args.no_save_mesh,
        limit=args.limit,
        debug=args.debug
    )