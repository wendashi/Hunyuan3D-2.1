#!/usr/bin/env python3
"""
检查训练数据的复杂度，诊断为什么训练后 vertices/faces 会减少
"""

import numpy as np
import glob
import os
import sys

def check_training_data(data_dir):
    """检查训练数据中 mesh 的复杂度"""
    print("=" * 60)
    print("检查训练数据复杂度")
    print("=" * 60)
    
    point_counts = []
    valid_files = 0
    
    pattern = os.path.join(data_dir, "*/geo_data/*_surface.npz")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 未找到数据文件: {pattern}")
        return
    
    print(f"找到 {len(files)} 个数据文件")
    print()
    
    for file_path in files:
        try:
            data = np.load(file_path)
            if 'random_surface' in data:
                random_surface = data['random_surface']
                point_counts.append(len(random_surface))
                valid_files += 1
        except Exception as e:
            print(f"警告: 无法加载 {file_path}: {e}")
    
    if not point_counts:
        print("❌ 没有有效的数据文件")
        return
    
    print(f"✅ 成功加载 {valid_files} 个数据文件")
    print()
    print("训练数据点数统计:")
    print(f"  平均: {np.mean(point_counts):.0f}")
    print(f"  中位数: {np.median(point_counts):.0f}")
    print(f"  最小: {np.min(point_counts)}")
    print(f"  最大: {np.max(point_counts)}")
    print(f"  标准差: {np.std(point_counts):.0f}")
    print()
    
    # 分析
    mean_points = np.mean(point_counts)
    median_points = np.median(point_counts)
    
    print("分析:")
    if mean_points < 50000:
        print(f"  ⚠️  训练数据点数较少（平均 {mean_points:.0f}），可能导致模型学习到简单的表示")
        print("  💡 建议：使用更高分辨率的训练数据，或增加 pc_size 参数")
    elif mean_points < 100000:
        print(f"  ⚠️  训练数据点数中等（平均 {mean_points:.0f}），可能不足以支持复杂 mesh")
        print("  💡 建议：检查是否需要更高分辨率的训练数据")
    else:
        print(f"  ✅ 训练数据点数充足（平均 {mean_points:.0f}）")
    
    print()
    
    # 检查配置中的 pc_size
    print("配置检查:")
    print("  当前配置 pc_size: 81920")
    if mean_points < 81920:
        print(f"  ⚠️  训练数据平均点数 ({mean_points:.0f}) < pc_size (81920)")
        print("  💡 这意味着部分数据会被重复采样，可能导致信息丢失")
    else:
        print(f"  ✅ 训练数据平均点数 ({mean_points:.0f}) >= pc_size (81920)")
    
    print()
    
    # 估算预期的 mesh 复杂度
    print("预期的 mesh 复杂度估算:")
    print("  octree_resolution = 256 → 网格大小 (257, 257, 257)")
    print("  → 理论上最多可提取的 vertices/faces 取决于:")
    print("     1. grid_logits 的质量（由 latent 决定）")
    print("     2. mc_level 阈值")
    print("     3. 实际有内容的区域大小")
    print()
    print("  如果训练数据点数较少，生成的 latent 可能不足以支持高复杂度 mesh")


def compare_meshes(mesh_before_path, mesh_after_path):
    """对比训练前后的 mesh"""
    try:
        import trimesh
        
        print("=" * 60)
        print("对比训练前后的 mesh")
        print("=" * 60)
        
        mesh_before = trimesh.load(mesh_before_path)
        mesh_after = trimesh.load(mesh_after_path)
        
        print(f"训练前 mesh:")
        print(f"  Vertices: {len(mesh_before.vertices)}")
        print(f"  Faces: {len(mesh_before.faces)}")
        print()
        
        print(f"训练后 mesh:")
        print(f"  Vertices: {len(mesh_after.vertices)}")
        print(f"  Faces: {len(mesh_after.faces)}")
        print()
        
        reduction_v = (1 - len(mesh_after.vertices) / len(mesh_before.vertices)) * 100
        reduction_f = (1 - len(mesh_after.faces) / len(mesh_before.faces)) * 100
        
        print(f"减少比例:")
        print(f"  Vertices: {reduction_v:.1f}%")
        print(f"  Faces: {reduction_f:.1f}%")
        print()
        
        # 检查 mesh 质量
        if hasattr(mesh_before, 'volume'):
            print(f"训练前 volume: {mesh_before.volume:.6f}")
        if hasattr(mesh_after, 'volume'):
            print(f"训练后 volume: {mesh_after.volume:.6f}")
            
    except ImportError:
        print("需要安装 trimesh: pip install trimesh")
    except Exception as e:
        print(f"无法加载 mesh: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  检查训练数据: python check_training_data.py <数据目录>")
        print("  对比 mesh: python check_training_data.py --compare <mesh1.glb> <mesh2.glb>")
        print()
        print("示例:")
        print("  python check_training_data.py /opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan")
        sys.exit(1)
    
    if sys.argv[1] == "--compare":
        if len(sys.argv) < 4:
            print("需要两个 mesh 文件路径")
            sys.exit(1)
        compare_meshes(sys.argv[2], sys.argv[3])
    else:
        data_dir = sys.argv[1]
        check_training_data(data_dir)

