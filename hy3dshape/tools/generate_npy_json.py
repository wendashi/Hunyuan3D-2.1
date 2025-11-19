#!/usr/bin/env python3
"""
生成 npy 文件的 JSON 索引
从指定目录扫描所有 .npy 文件，读取对应的 _parts.json 元数据，生成汇总 JSON
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def detect_dataset_from_filename(filename: str) -> str:
    """
    根据文件名自动判断数据集类型
    
    Args:
        filename: 文件名（不含路径）
    
    Returns:
        数据集名称: "DiFa-LowPoly", "DiFa-HighPoly", 或 "Cloth4D"
    """
    filename_lower = filename.lower()
    if "lowpoly" in filename_lower:
        return "DiFa-LowPoly"
    elif "highpoly" in filename_lower:
        return "DiFa-HighPoly"
    else:
        return "Cloth4D"


def generate_npy_json(
    npy_dir: str,
    output_json: str,
    dataset: str = None,
    mesh_dir: str = None,
) -> None:
    """
    生成 npy 文件的 JSON 索引
    
    Args:
        npy_dir: 包含 .npy 文件的目录
        output_json: 输出的 JSON 文件路径
        dataset: 数据集名称
        mesh_dir: 原始 mesh 文件目录（如果 _parts.json 中没有 input 字段，则从该目录查找）
    """
    npy_dir = Path(npy_dir)
    if not npy_dir.exists():
        raise FileNotFoundError(f"目录不存在: {npy_dir}")
    
    json_data: List[Dict] = []
    
    # 扫描所有 .npy 文件
    npy_files = sorted(npy_dir.glob("*.npy"))
    
    print(f"[INFO] 找到 {len(npy_files)} 个 .npy 文件")
    
    for npy_file in npy_files:
        # 获取文件名（不含扩展名）
        basename = npy_file.stem
        glb_filename = f"{basename}.glb"
        
        # 根据文件名自动判断数据集类型
        detected_dataset = detect_dataset_from_filename(glb_filename)
        entry_dataset = dataset if dataset else detected_dataset
        
        # 读取对应的 _parts.json 文件
        parts_json_path = npy_dir / f"{basename}_parts.json"
        
        num_parts = 0
        mesh_path = None
        
        if parts_json_path.exists():
            try:
                with open(parts_json_path, 'r', encoding='utf-8') as f:
                    parts_data = json.load(f)
                    num_parts = parts_data.get('parts_processed', 0)
                    # 从 _parts.json 中获取原始 mesh 路径
                    mesh_path = parts_data.get('input')
            except Exception as e:
                print(f"[WARNING] 无法读取 {parts_json_path}: {e}")
        
        # 如果 _parts.json 中没有 input 字段，尝试从 mesh_dir 查找
        if not mesh_path and mesh_dir:
            mesh_file = Path(mesh_dir) / glb_filename
            if mesh_file.exists():
                mesh_path = str(mesh_file)
        
        # 如果还是找不到，使用默认路径
        if not mesh_path:
            # 尝试从 npy 文件名推断原始路径
            # 假设原始文件在 raw-data 目录
            raw_data_dir = npy_dir.parent / "raw-data"
            if raw_data_dir.exists():
                mesh_file = raw_data_dir / glb_filename
                if mesh_file.exists():
                    mesh_path = str(mesh_file)
        
        # 构建 JSON 条目（使用绝对路径）
        entry = {
            "file": glb_filename,
            "num_parts": num_parts,
            "mesh_path": mesh_path or f"未知路径/{glb_filename}",
            "surface_path": str(npy_file.resolve()),  # 使用绝对路径
            "dataset": entry_dataset,
        }
        
        json_data.append(entry)
    
    # 按文件名排序
    json_data.sort(key=lambda x: x['file'])
    
    # 保存 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"[INFO] 成功生成 JSON 文件: {output_json}")
    print(f"[INFO] 包含 {len(json_data)} 个条目")
    
    # 统计信息
    total_parts = sum(entry['num_parts'] for entry in json_data)
    missing_mesh = sum(1 for entry in json_data if '未知路径' in entry['mesh_path'])
    
    print(f"[INFO] 总部件数: {total_parts}")
    if missing_mesh > 0:
        print(f"[WARNING] {missing_mesh} 个文件未找到原始 mesh 路径")


def main():
    parser = argparse.ArgumentParser(
        description="生成 npy 文件的 JSON 索引"
    )
    parser.add_argument(
        "--npy_dir",
        type=str,
        required=True,
        help="包含 .npy 文件的目录路径"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="输出的 JSON 文件路径"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称（可选，如果不指定则根据文件名自动判断：LowPoly->DiFa-LowPoly, HighPoly->DiFa-HighPoly, 其他->Cloth4D）"
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=None,
        help="原始 mesh 文件目录（可选，如果 _parts.json 中没有 input 字段则使用）"
    )
    
    args = parser.parse_args()
    
    generate_npy_json(
        npy_dir=args.npy_dir,
        output_json=args.output_json,
        dataset=args.dataset,
        mesh_dir=args.mesh_dir,
    )


if __name__ == "__main__":
    main()

