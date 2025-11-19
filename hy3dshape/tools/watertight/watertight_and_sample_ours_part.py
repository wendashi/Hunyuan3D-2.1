import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import trimesh

# 复用 watertight_and_sample_ours.py 中的核心实现
from watertight_and_sample_ours import (
    Watertight,
    random_sample_pointcloud,
)


@dataclass
class MeshPart:
    name: str
    mesh: trimesh.Trimesh
    material_names: List[str]

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices)

    @property
    def bounding_box_volume(self) -> float:
        if self.mesh.bounding_box_oriented.extents is None:
            return 0.0
        extents = self.mesh.bounding_box_oriented.extents
        return float(np.prod(extents))


def _extract_mesh_parts(scene: trimesh.Scene) -> List[MeshPart]:
    parts: List[MeshPart] = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geometry = scene.geometry.get(geom_name)
        if geometry is None:
            continue
        if not isinstance(geometry, trimesh.Trimesh):
            try:
                geometry = geometry.as_trimesh()
            except Exception:
                continue
        mesh = geometry.copy()
        if transform is not None:
            mesh.apply_transform(transform)

        material_names: List[str] = []
        visual = getattr(geometry, "visual", None)
        if visual is not None:
            material = getattr(visual, "material", None)
            if material is not None and hasattr(material, "name"):
                material_names.append(material.name)
            elif hasattr(visual, "materials") and visual.materials:
                for mat in visual.materials:
                    if mat is not None and hasattr(mat, "name"):
                        material_names.append(mat.name)

        part_name = node_name or geom_name or f"part_{len(parts)}"
        parts.append(MeshPart(name=part_name, mesh=mesh, material_names=material_names))

    if not parts and isinstance(scene, trimesh.Scene):
        combined = scene.dump(concatenate=True)
        if isinstance(combined, list):
            for idx, mesh in enumerate(combined):
                parts.append(MeshPart(name=f"part_{idx}", mesh=mesh, material_names=[]))
        elif isinstance(combined, trimesh.Trimesh):
            parts.append(MeshPart(name="part_0", mesh=combined, material_names=[]))

    return parts


def _split_mesh_by_material(mesh: trimesh.Trimesh) -> List[MeshPart]:
    if mesh.visual.kind != "face":
        return [MeshPart(name="part_0", mesh=mesh, material_names=[])]

    face_materials = getattr(mesh.visual, "face_materials", None)
    materials = getattr(mesh.visual, "materials", None)

    if face_materials is None or materials is None:
        return [MeshPart(name="part_0", mesh=mesh, material_names=[])]

    unique_material_ids = np.unique(face_materials)
    parts: List[MeshPart] = []

    for mat_id in unique_material_ids:
        mask = face_materials == mat_id
        if not np.any(mask):
            continue
        submesh = mesh.submesh([np.nonzero(mask)[0]], append=True, repair=False)
        material_name = ""
        if 0 <= mat_id < len(materials):
            material = materials[mat_id]
            if material is not None and hasattr(material, "name"):
                material_name = material.name
        part_name = material_name or f"material_{mat_id}"
        parts.append(MeshPart(name=part_name, mesh=submesh, material_names=[material_name]))

    if not parts:
        parts.append(MeshPart(name="part_0", mesh=mesh, material_names=[]))

    return parts


def load_mesh_parts(mesh_path: str) -> List[MeshPart]:
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        parts = _extract_mesh_parts(loaded)
        if parts:
            return parts

    if isinstance(loaded, trimesh.Trimesh):
        return _split_mesh_by_material(loaded)

    if isinstance(loaded, list):
        parts: List[MeshPart] = []
        for idx, item in enumerate(loaded):
            if isinstance(item, trimesh.Trimesh):
                parts.append(MeshPart(name=f"part_{idx}", mesh=item, material_names=[]))
        return parts

    raise RuntimeError(f"无法解析网格: {mesh_path}")


def normalize_parts(parts: List[MeshPart]) -> Tuple[np.ndarray, float]:
    """
    归一化所有部件到 [-1, 1]^3 范围，保持各部件相对位置不变。
    参考 normalize_to_unit_box 的实现，确保整体 object 的范围在 x,y,z 轴从 -1 到 1。
    
    Returns:
        center: 归一化前的中心点
        scale: 归一化使用的尺度因子
    """
    if not parts:
        raise ValueError("没有可归一化的部件")

    # 收集所有部件的顶点
    all_vertices = np.concatenate([part.mesh.vertices for part in parts if len(part.mesh.vertices) > 0], axis=0)
    if all_vertices.size == 0:
        raise ValueError("部件网格没有顶点")

    # 转换为 float64 以确保精度
    all_vertices = np.asarray(all_vertices, dtype=np.float64)
    
    # 计算整体边界
    V_min = all_vertices.min(axis=0)
    V_max = all_vertices.max(axis=0)
    
    # 计算中心点
    center = (V_min + V_max) / 2.0
    
    # 计算尺度：除以2是因为范围是2（从-1到1），乘以1.01是为了留出一点边距
    scale = (V_max - V_min).max() / 2.0 * 1.01
    if math.isclose(scale, 0.0):
        scale = 1.0

    # 对所有部件应用相同的归一化变换，保持相对位置
    for part in parts:
        if len(part.mesh.vertices) == 0:
            continue
        # 转换为 float64 进行计算
        V = np.asarray(part.mesh.vertices, dtype=np.float64)
        # 应用归一化：先平移到中心，再缩放
        V_normalized = (V - center) / scale
        # 转换回原始数据类型（通常是 float64）
        part.mesh.vertices = V_normalized

    return center, scale


def should_filter_part(
    part: MeshPart,
    trim_keywords: List[str],
    min_faces: int,
    min_volume: float,
    skip_trim_filter: bool,
) -> bool:
    if part.face_count == 0 or part.vertex_count == 0:
        return True

    if not skip_trim_filter:
        name_lower = part.name.lower()
        if any(keyword in name_lower for keyword in trim_keywords):
            return True

        for mat_name in part.material_names:
            if mat_name and any(keyword in mat_name.lower() for keyword in trim_keywords):
                return True

    if part.face_count < min_faces:
        return True

    if part.bounding_box_volume < min_volume:
        return True

    return False


def filter_parts(
    parts: List[MeshPart],
    min_faces: int,
    min_volume: float,
    skip_trim_filter: bool,
    trim_keywords: Optional[List[str]] = None,
) -> List[MeshPart]:
    keywords = trim_keywords or ["trim", "bindedtrim", "topstitch", "edging", "delete me"]
    filtered: List[MeshPart] = []

    for part in parts:
        if should_filter_part(
            part,
            trim_keywords=keywords,
            min_faces=min_faces,
            min_volume=min_volume,
            skip_trim_filter=skip_trim_filter,
        ):
            continue
        filtered.append(part)

    return filtered


def process_part(
    part: MeshPart,
    epsilon: float,
    grid_res: int,
    sample_num: int = 124928,
) -> Tuple[np.ndarray, np.ndarray]:
    """处理单个部件：水密化并采样点云
    
    Returns:
        surface_points: (sample_num, 3) float16 array
        surface_normals: (sample_num, 3) float16 array
    """
    V = np.asarray(part.mesh.vertices, dtype=np.float64)
    F = np.asarray(part.mesh.faces, dtype=np.int64)

    # 水密化处理
    mc_verts, mc_faces = Watertight(V, F, epsilon=epsilon, grid_res=grid_res)
    
    # 创建 trimesh 对象用于采样
    watertight_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces, process=False)
    
    # 采样固定数量的点
    surface_points, surface_normals = random_sample_pointcloud(watertight_mesh, num=sample_num)
    
    # 转换为 float16
    surface_points = surface_points.astype(np.float16)
    surface_normals = surface_normals.astype(np.float16)
    
    return surface_points, surface_normals


def aggregate_data(
    object_points: np.ndarray,
    object_normals: np.ndarray,
    parts_points: List[np.ndarray],
    parts_normals: List[np.ndarray],
) -> dict:
    """聚合数据为 npy 格式
    
    Returns:
        dict with 'object' and 'parts' keys
    """
    # 构建 parts 列表
    parts_data = []
    for idx, (points, normals) in enumerate(zip(parts_points, parts_normals)):
        parts_data.append({
            'index': idx,
            'surface_points': points.astype(np.float16),
            'surface_normals': normals.astype(np.float16),
        })
    
    # 构建最终数据结构
    result = {
        'object': {
            'surface_points': object_points.astype(np.float16),
            'surface_normals': object_normals.astype(np.float16),
        },
        'parts': parts_data,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="按部件拆分 GLB 文件，执行 watertight 与采样，输出 npy 格式点云数据"
    )
    parser.add_argument("--input_obj", type=str, required=True, help="输入 GLB 文件路径")
    parser.add_argument("--output_prefix", type=str, required=True, help="输出文件前缀（不含扩展名）")
    parser.add_argument("--grid_res", type=int, default=256, help="Marching cubes 网格分辨率")
    parser.add_argument("--epsilon", type=float, default=2.0 / 256, help="Marching cubes 等值面 epsilon")
    parser.add_argument("--sample_num", type=int, default=124928, help="每个部件和整体的采样点数")
    parser.add_argument("--min_faces", type=int, default=5, help="保留部件的最小面数阈值")
    parser.add_argument("--min_volume", type=float, default=7e-6, help="保留部件的最小包围盒体积")
    parser.add_argument("--max_parts", type=int, default=30, help="最多处理的部件数量，按面数排序保留")
    parser.add_argument("--skip_trim_filter", action="store_true", help="跳过 trim 关键字过滤")
    parser.add_argument(
        "--trim_keywords",
        type=str,
        nargs="*",
        default=None,
        help="自定义 trim 过滤关键字，默认 ['trim','bindedtrim','topstitch','edging','delete me']",
    )
    parser.add_argument("--metadata_path", type=str, default=None, help="输出部件元数据 JSON 路径")
    args = parser.parse_args()

    input_path = args.input_obj
    output_prefix = args.output_prefix

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 加载并拆分部件
    parts = load_mesh_parts(input_path)
    if not parts:
        raise RuntimeError("未从输入网格中提取到任何部件")

    # 归一化所有部件到 [-1,1]^3，保持各部件相对位置不变
    normalize_parts(parts)

    # 过滤部件
    filtered_parts = filter_parts(
        parts,
        min_faces=args.min_faces,
        min_volume=args.min_volume,
        skip_trim_filter=args.skip_trim_filter,
        trim_keywords=args.trim_keywords,
    )

    if not filtered_parts:
        raise RuntimeError("所有部件均被过滤，请调整阈值参数")

    filtered_parts.sort(key=lambda p: p.face_count, reverse=True)
    if args.max_parts > 0:
        filtered_parts = filtered_parts[: args.max_parts]

    # 处理每个部件
    parts_points: List[np.ndarray] = []
    parts_normals: List[np.ndarray] = []
    part_summaries = []

    parent_folder = os.path.dirname(output_prefix)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    for idx, part in enumerate(filtered_parts):
        print(f"[INFO] Processing part {idx + 1}/{len(filtered_parts)}: {part.name} ({part.face_count} faces)")
        try:
            surface_points, surface_normals = process_part(
                part,
                epsilon=args.epsilon,
                grid_res=args.grid_res,
                sample_num=args.sample_num,
            )
            parts_points.append(surface_points)
            parts_normals.append(surface_normals)

            part_summary = {
                "name": part.name,
                "face_count": part.face_count,
                "vertex_count": part.vertex_count,
                "material_names": part.material_names,
                "bounding_box_volume": part.bounding_box_volume,
            }
            part_summaries.append(part_summary)
        except Exception as exc:
            print(f"[WARNING] Failed to process part {part.name}: {exc}")

    if not parts_points:
        raise RuntimeError("未成功处理任何部件")

    # 处理整体 mesh：合并所有部件
    print(f"[INFO] Processing combined object mesh...")
    try:
        # 合并所有部件的顶点和面
        all_vertices = []
        all_faces = []
        face_offset = 0
        
        for part in filtered_parts:
            if len(part.mesh.vertices) == 0 or len(part.mesh.faces) == 0:
                continue
            all_vertices.append(part.mesh.vertices)
            all_faces.append(part.mesh.faces + face_offset)
            face_offset += len(part.mesh.vertices)
        
        if all_vertices:
            combined_vertices = np.concatenate(all_vertices, axis=0)
            combined_faces = np.concatenate(all_faces, axis=0)
            
            # 对整体进行 watertight 和采样
            V = np.asarray(combined_vertices, dtype=np.float64)
            F = np.asarray(combined_faces, dtype=np.int64)
            mc_verts, mc_faces = Watertight(V, F, epsilon=args.epsilon, grid_res=args.grid_res)
            watertight_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces, process=False)
            object_points, object_normals = random_sample_pointcloud(watertight_mesh, num=args.sample_num)
        else:
            raise RuntimeError("无法合并部件顶点")
    except Exception as exc:
        print(f"[WARNING] Failed to process combined object: {exc}")
        raise

    # 聚合数据
    result_data = aggregate_data(
        object_points=object_points,
        object_normals=object_normals,
        parts_points=parts_points,
        parts_normals=parts_normals,
    )

    # 保存 npy 文件
    output_file = f"{output_prefix}.npy"
    np.save(output_file, result_data)
    print(f"[INFO] Saved point cloud data: {output_file}")

    # 保存元数据
    parts_json_path = args.metadata_path or f"{output_prefix}_parts.json"
    with open(parts_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": input_path,
                "parts_processed": len(part_summaries),
                "grid_res": args.grid_res,
                "epsilon": args.epsilon,
                "sample_num": args.sample_num,
                "min_faces": args.min_faces,
                "min_volume": args.min_volume,
                "skip_trim_filter": args.skip_trim_filter,
                "parts": part_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[INFO] Saved parts metadata: {parts_json_path}")


if __name__ == "__main__":
    main()

