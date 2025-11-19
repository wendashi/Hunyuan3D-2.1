import json
import os
import re
import numpy as np
import trimesh
from collections import defaultdict
from tqdm import tqdm

# 配置路径
GT_JSON = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train-Test-Set/PartCrafter/test/merged-test<16.json'
INFER_BASE_PATH = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/PartCrafter/merged-test-1027/20251027_22_25_32'

# 输出文件
OUTPUT_FILE = '/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/3d-metrics/coordinate_analysis_detailed.txt'


def extract_filename_from_mesh_path(mesh_path):
    """从mesh_path提取文件名，并转换为infer路径格式"""
    filename = os.path.basename(mesh_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    match = re.search(r'(\d+)', name_without_ext)
    if match:
        num = int(match.group(1))
        # 根据前缀决定补0位数：LowPoly用4位，HighPoly和其他用5位
        if name_without_ext.startswith('LowPoly'):
            num_str = f"{num:04d}"  # LowPoly: 4位数字
        else:
            num_str = f"{num:05d}"  # HighPoly等: 5位数字
        name_without_ext = re.sub(r'\d+', num_str, name_without_ext, count=1)
    
    return name_without_ext


def build_infer_path(mesh_path, num_parts, infer_base_path):
    """构建infer结果的object.glb路径"""
    folder_name = extract_filename_from_mesh_path(mesh_path)
    folder_name = f"{folder_name}_parts_{num_parts}"
    infer_path = os.path.join(infer_base_path, folder_name, 'object.glb')
    return infer_path


def load_glb_mesh(glb_path):
    """加载GLB文件并返回trimesh对象"""
    if not os.path.exists(glb_path):
        return None
    
    try:
        mesh = trimesh.load(glb_path)
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if len(geometries) == 1:
                mesh = geometries[0]
            else:
                mesh = trimesh.util.concatenate(geometries)
        return mesh
    except Exception as e:
        return None


def analyze_mesh(mesh):
    """分析mesh的坐标信息"""
    if mesh is None:
        return None
    
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    bounds_size = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2
    center_distance = np.linalg.norm(center)
    
    vertices = mesh.vertices
    vertex_center = np.mean(vertices, axis=0)
    vertex_center_distance = np.linalg.norm(vertex_center)
    
    return {
        'bounds_min': bounds_min,
        'bounds_max': bounds_max,
        'bounds_size': bounds_size,
        'center': center,
        'center_distance': center_distance,
        'vertex_center': vertex_center,
        'vertex_center_distance': vertex_center_distance,
        'max_dimension': np.max(bounds_size),
        'min_dimension': np.min(bounds_size),
    }


def main():
    print("Loading GT JSON...")
    with open(GT_JSON, 'r') as f:
        gt_data = json.load(f)
    
    print(f"Found {len(gt_data)} samples\n")
    
    # 分析指定数量的样本
    num_samples = min(100, len(gt_data))  # 分析100个样本
    print(f"Analyzing {num_samples} samples...\n")
    
    results = []
    failed_samples = []
    
    for i in tqdm(range(num_samples), desc="Processing samples"):
        sample = gt_data[i]
        gt_mesh_path = sample['mesh_path']
        num_parts = sample['num_parts']
        dataset = sample.get('dataset', 'unknown')
        file_name = os.path.basename(gt_mesh_path)
        
        infer_mesh_path = build_infer_path(gt_mesh_path, num_parts, INFER_BASE_PATH)
        
        gt_mesh = load_glb_mesh(gt_mesh_path)
        infer_mesh = load_glb_mesh(infer_mesh_path)
        
        # 详细记录失败原因
        if gt_mesh is None or infer_mesh is None:
            failure_reason = []
            if not os.path.exists(gt_mesh_path):
                failure_reason.append(f"GT path not exists: {gt_mesh_path}")
            elif gt_mesh is None:
                failure_reason.append(f"GT mesh load failed")
            
            if not os.path.exists(infer_mesh_path):
                failure_reason.append(f"Infer path not exists: {infer_mesh_path}")
            elif infer_mesh is None:
                failure_reason.append(f"Infer mesh load failed")
            
            failed_samples.append({
                'file': file_name,
                'gt_path': gt_mesh_path,
                'infer_path': infer_mesh_path,
                'reason': '; '.join(failure_reason) if failure_reason else 'Unknown'
            })
            continue
        
        gt_info = analyze_mesh(gt_mesh)
        infer_info = analyze_mesh(infer_mesh)
        
        center_diff = infer_info['center'] - gt_info['center']
        size_diff = infer_info['bounds_size'] - gt_info['bounds_size']
        
        results.append({
            'sample_idx': i,
            'file': file_name,
            'dataset': dataset,
            'num_parts': num_parts,
            'gt_info': gt_info,
            'infer_info': infer_info,
            'center_diff': center_diff,
            'size_diff': size_diff,
        })
    
    print(f"\nSuccessfully analyzed {len(results)} samples")
    if failed_samples:
        print(f"\nFailed to load {len(failed_samples)} samples")
        print("\nFailure Analysis:")
        print("-"*80)
        
        # 统计失败原因
        failure_reasons = {}
        path_not_exists_gt = 0
        path_not_exists_infer = 0
        load_failed_gt = 0
        load_failed_infer = 0
        
        for failed in failed_samples:
            reason = failed['reason']
            if 'GT path not exists' in reason:
                path_not_exists_gt += 1
            if 'Infer path not exists' in reason:
                path_not_exists_infer += 1
            if 'GT mesh load failed' in reason:
                load_failed_gt += 1
            if 'Infer mesh load failed' in reason:
                load_failed_infer += 1
        
        print(f"  GT path not exists: {path_not_exists_gt}")
        print(f"  Infer path not exists: {path_not_exists_infer}")
        print(f"  GT mesh load failed: {load_failed_gt}")
        print(f"  Infer mesh load failed: {load_failed_infer}")
        
        # 显示前10个失败样本的详细信息
        print(f"\n  First 10 failed samples:")
        for i, failed in enumerate(failed_samples[:10]):
            print(f"    {i+1}. {failed['file']}")
            print(f"       Reason: {failed['reason']}")
            if 'Infer path not exists' in failed['reason']:
                print(f"       Infer path: {failed['infer_path']}")
    
    # 统计分析
    print("\n" + "="*80)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*80)
    
    if len(results) == 0:
        print("No valid samples to analyze!")
        return
    
    # 提取所有数据
    gt_centers = np.array([r['gt_info']['center'] for r in results])
    infer_centers = np.array([r['infer_info']['center'] for r in results])
    center_diffs = np.array([r['center_diff'] for r in results])
    
    gt_sizes = np.array([r['gt_info']['bounds_size'] for r in results])
    infer_sizes = np.array([r['infer_info']['bounds_size'] for r in results])
    size_diffs = np.array([r['size_diff'] for r in results])
    
    gt_center_distances = np.array([r['gt_info']['center_distance'] for r in results])
    infer_center_distances = np.array([r['infer_info']['center_distance'] for r in results])
    
    # 1. 中心位置分析
    print("\n1. CENTER POSITION ANALYSIS")
    print("-"*80)
    print(f"GT Center Statistics (X, Y, Z):")
    print(f"  Mean:   ({gt_centers[:, 0].mean():.4f}, {gt_centers[:, 1].mean():.4f}, {gt_centers[:, 2].mean():.4f})")
    print(f"  Std:    ({gt_centers[:, 0].std():.4f}, {gt_centers[:, 1].std():.4f}, {gt_centers[:, 2].std():.4f})")
    print(f"  Min:    ({gt_centers[:, 0].min():.4f}, {gt_centers[:, 1].min():.4f}, {gt_centers[:, 2].min():.4f})")
    print(f"  Max:    ({gt_centers[:, 0].max():.4f}, {gt_centers[:, 1].max():.4f}, {gt_centers[:, 2].max():.4f})")
    
    print(f"\nInfer Center Statistics (X, Y, Z):")
    print(f"  Mean:   ({infer_centers[:, 0].mean():.4f}, {infer_centers[:, 1].mean():.4f}, {infer_centers[:, 2].mean():.4f})")
    print(f"  Std:    ({infer_centers[:, 0].std():.4f}, {infer_centers[:, 1].std():.4f}, {infer_centers[:, 2].std():.4f})")
    print(f"  Min:    ({infer_centers[:, 0].min():.4f}, {infer_centers[:, 1].min():.4f}, {infer_centers[:, 2].min():.4f})")
    print(f"  Max:    ({infer_centers[:, 0].max():.4f}, {infer_centers[:, 1].max():.4f}, {infer_centers[:, 2].max():.4f})")
    
    print(f"\nCenter Offset Statistics (Infer - GT):")
    print(f"  Mean:   ({center_diffs[:, 0].mean():.4f}, {center_diffs[:, 1].mean():.4f}, {center_diffs[:, 2].mean():.4f})")
    print(f"  Std:    ({center_diffs[:, 0].std():.4f}, {center_diffs[:, 1].std():.4f}, {center_diffs[:, 2].std():.4f})")
    print(f"  Abs Mean: ({np.abs(center_diffs[:, 0]).mean():.4f}, {np.abs(center_diffs[:, 1]).mean():.4f}, {np.abs(center_diffs[:, 2]).mean():.4f})")
    
    # Y轴特别分析（因为差异最大）
    print(f"\n  Y-Axis Analysis (largest difference):")
    print(f"    GT Y mean: {gt_centers[:, 1].mean():.4f} ± {gt_centers[:, 1].std():.4f}")
    print(f"    Infer Y mean: {infer_centers[:, 1].mean():.4f} ± {infer_centers[:, 1].std():.4f}")
    print(f"    Y offset mean: {center_diffs[:, 1].mean():.4f} ± {center_diffs[:, 1].std():.4f}")
    print(f"    Y offset abs mean: {np.abs(center_diffs[:, 1]).mean():.4f}")
    
    # 2. 尺寸分析
    print("\n2. SIZE ANALYSIS")
    print("-"*80)
    print(f"GT Size Statistics (X, Y, Z):")
    print(f"  Mean:   ({gt_sizes[:, 0].mean():.4f}, {gt_sizes[:, 1].mean():.4f}, {gt_sizes[:, 2].mean():.4f})")
    print(f"  Std:    ({gt_sizes[:, 0].std():.4f}, {gt_sizes[:, 1].std():.4f}, {gt_sizes[:, 2].std():.4f})")
    
    print(f"\nInfer Size Statistics (X, Y, Z):")
    print(f"  Mean:   ({infer_sizes[:, 0].mean():.4f}, {infer_sizes[:, 1].mean():.4f}, {infer_sizes[:, 2].mean():.4f})")
    print(f"  Std:    ({infer_sizes[:, 0].std():.4f}, {infer_sizes[:, 1].std():.4f}, {infer_sizes[:, 2].std():.4f})")
    
    print(f"\nSize Difference Statistics (Infer - GT):")
    print(f"  Mean:   ({size_diffs[:, 0].mean():.4f}, {size_diffs[:, 1].mean():.4f}, {size_diffs[:, 2].mean():.4f})")
    print(f"  Std:    ({size_diffs[:, 0].std():.4f}, {size_diffs[:, 1].std():.4f}, {size_diffs[:, 2].std():.4f})")
    
    # 尺寸比例分析
    size_ratios = infer_sizes / (gt_sizes + 1e-8)  # 避免除零
    print(f"\nSize Ratio Statistics (Infer / GT):")
    print(f"  Mean:   ({size_ratios[:, 0].mean():.4f}, {size_ratios[:, 1].mean():.4f}, {size_ratios[:, 2].mean():.4f})")
    print(f"  Std:    ({size_ratios[:, 0].std():.4f}, {size_ratios[:, 1].std():.4f}, {size_ratios[:, 2].std():.4f})")
    
    # 3. 中心距离分析
    print("\n3. CENTER DISTANCE FROM ORIGIN")
    print("-"*80)
    print(f"GT Center Distance:")
    print(f"  Mean: {gt_center_distances.mean():.4f}")
    print(f"  Std:  {gt_center_distances.std():.4f}")
    print(f"  Min:  {gt_center_distances.min():.4f}")
    print(f"  Max:  {gt_center_distances.max():.4f}")
    
    print(f"\nInfer Center Distance:")
    print(f"  Mean: {infer_center_distances.mean():.4f}")
    print(f"  Std:  {infer_center_distances.std():.4f}")
    print(f"  Min:  {infer_center_distances.min():.4f}")
    print(f"  Max:  {infer_center_distances.max():.4f}")
    
    # 4. 按数据集分组分析
    print("\n4. ANALYSIS BY DATASET")
    print("-"*80)
    dataset_groups = defaultdict(list)
    for r in results:
        dataset_groups[r['dataset']].append(r)
    
    for dataset, group_results in dataset_groups.items():
        group_gt_centers = np.array([r['gt_info']['center'] for r in group_results])
        group_infer_centers = np.array([r['infer_info']['center'] for r in group_results])
        group_center_diffs = np.array([r['center_diff'] for r in group_results])
        
        print(f"\n{dataset} ({len(group_results)} samples):")
        print(f"  GT Center Mean: ({group_gt_centers[:, 0].mean():.4f}, {group_gt_centers[:, 1].mean():.4f}, {group_gt_centers[:, 2].mean():.4f})")
        print(f"  Infer Center Mean: ({group_infer_centers[:, 0].mean():.4f}, {group_infer_centers[:, 1].mean():.4f}, {group_infer_centers[:, 2].mean():.4f})")
        print(f"  Y Offset Mean: {group_center_diffs[:, 1].mean():.4f} ± {group_center_diffs[:, 1].std():.4f}")
    
    # 5. 关键发现和建议
    print("\n5. KEY FINDINGS AND RECOMMENDATIONS")
    print("-"*80)
    
    y_offset_mean = center_diffs[:, 1].mean()
    y_offset_std = center_diffs[:, 1].std()
    
    print(f"✓ GT meshes are consistently offset in Y-axis:")
    print(f"    Average Y position: {gt_centers[:, 1].mean():.4f}")
    print(f"    This suggests GT uses a different coordinate system origin")
    
    print(f"\n✓ Infer meshes are closer to origin:")
    print(f"    Average Y position: {infer_centers[:, 1].mean():.4f}")
    print(f"    Standard deviation: {infer_centers[:, 1].std():.4f}")
    
    print(f"\n✓ Average Y-axis offset: {y_offset_mean:.4f} ± {y_offset_std:.4f}")
    print(f"    This is the main difference between GT and Infer coordinate systems")
    
    size_ratio_mean = size_ratios.mean(axis=0)
    size_ratio_std = size_ratios.std(axis=0)
    
    print(f"\n✓ Size ratios vary significantly:")
    print(f"    X: {size_ratio_mean[0]:.4f} ± {size_ratio_std[0]:.4f}")
    print(f"    Y: {size_ratio_mean[1]:.4f} ± {size_ratio_std[1]:.4f}")
    print(f"    Z: {size_ratio_mean[2]:.4f} ± {size_ratio_std[2]:.4f}")
    print(f"    This suggests inconsistent scaling between GT and Infer")
    
    print(f"\n📌 RECOMMENDATIONS:")
    print(f"  1. Enable center alignment (align_center=True) in calculate_metrics.py")
    print(f"     This will move both meshes to origin before computing metrics")
    print(f"  2. Consider scale alignment (align_scale=True) if you want to evaluate")
    print(f"     shape similarity regardless of size differences")
    print(f"  3. The Y-axis offset is consistent, suggesting a systematic difference")
    print(f"     in coordinate system origins between GT and Infer")
    
    # 保存详细报告
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("Detailed Coordinate Analysis Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples analyzed: {len(results)}\n")
        f.write(f"Failed samples: {len(failed_samples)}\n\n")
        
        if failed_samples:
            f.write("Failure Analysis:\n")
            f.write("-"*80 + "\n")
            path_not_exists_infer = sum(1 for fs in failed_samples if 'Infer path not exists' in fs['reason'])
            path_not_exists_gt = sum(1 for fs in failed_samples if 'GT path not exists' in fs['reason'])
            load_failed_infer = sum(1 for fs in failed_samples if 'Infer mesh load failed' in fs['reason'])
            load_failed_gt = sum(1 for fs in failed_samples if 'GT mesh load failed' in fs['reason'])
            
            f.write(f"  GT path not exists: {path_not_exists_gt}\n")
            f.write(f"  Infer path not exists: {path_not_exists_infer}\n")
            f.write(f"  GT mesh load failed: {load_failed_gt}\n")
            f.write(f"  Infer mesh load failed: {load_failed_infer}\n\n")
            
            f.write("  Sample failed paths (first 20):\n")
            for i, failed in enumerate(failed_samples[:20]):
                f.write(f"    {i+1}. {failed['file']}: {failed['reason']}\n")
                if 'Infer path not exists' in failed['reason']:
                    f.write(f"       Path: {failed['infer_path']}\n")
            f.write("\n")
        
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"GT Center Mean: ({gt_centers[:, 0].mean():.4f}, {gt_centers[:, 1].mean():.4f}, {gt_centers[:, 2].mean():.4f})\n")
        f.write(f"Infer Center Mean: ({infer_centers[:, 0].mean():.4f}, {infer_centers[:, 1].mean():.4f}, {infer_centers[:, 2].mean():.4f})\n")
        f.write(f"Y-axis Offset Mean: {y_offset_mean:.4f} ± {y_offset_std:.4f}\n")
        f.write(f"Size Ratio Mean: ({size_ratio_mean[0]:.4f}, {size_ratio_mean[1]:.4f}, {size_ratio_mean[2]:.4f})\n")
    
    print(f"\n\nDetailed report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

