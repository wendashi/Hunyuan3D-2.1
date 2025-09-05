# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import os
import numpy as np
from scipy.stats import truncnorm
import trimesh

# libigl (python binding)
import igl


def read_mesh_vertices_faces(mesh_path: str):
    """Try reading a triangle mesh using libigl; if it fails, fall back to trimesh."""
    V = None
    F = None
    try:
        V, F = igl.read_triangle_mesh(mesh_path)
        if V is not None and F is not None and V.size > 0 and F.size > 0:
            V = np.asarray(V, dtype=np.float64)
            F = np.asarray(F, dtype=np.int64)
            return V, F
    except Exception:
        pass

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh as Trimesh: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {mesh_path}")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    return V, F


def call_signed_distance(points: np.ndarray, V: np.ndarray, F: np.ndarray, sign_type=None) -> np.ndarray:
    """Version-compatible wrapper for igl.signed_distance.

    Always returns only the SDF array regardless of how many values igl returns.
    """
    points = np.asarray(points, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if sign_type is not None:
        try:
            ret = igl.signed_distance(points, V, F, sign_type=sign_type)
            return ret[0]
        except TypeError:
            pass
    ret = igl.signed_distance(points, V, F)
    return ret[0]


def random_sample_pointcloud(mesh: trimesh.Trimesh, num: int = 30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]


def sharp_sample_pointcloud(mesh: trimesh.Trimesh, num: int = 16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = (sharp_mask[edge_a] * sharp_mask[edge_b])
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights_sum = np.sum(weights)
    if weights_sum <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= weights_sum

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number, side='right')
    index = np.clip(index, 0, len(weights) - 1)
    samples = w * sharp_verts_a[index] + (1.0 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1.0 - w) * sharp_verts_bn[index]
    return samples, normals


def sample_sdf(mesh: trimesh.Trimesh, random_surface: np.ndarray, sharp_surface: np.ndarray):
    n_volume_points = sharp_surface.shape[0] * 2
    vol_points = (np.random.rand(n_volume_points, 3) - 0.5) * 2 * 1.05

    a, b = -0.25, 0.25
    mu = 0

    offset1 = truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(len(random_surface), 3))
    offset2 = truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05, size=(len(random_surface), 3))
    random_near_points = np.concatenate([
        random_surface + offset1,
        random_surface + offset2
    ], axis=0)

    unit_num = max(1, len(sharp_surface) // 6)
    sharp_near_points = np.concatenate([
        sharp_surface[:unit_num] + np.random.normal(scale=0.001, size=(unit_num, 3)),
        sharp_surface[unit_num:unit_num * 2] + np.random.normal(scale=0.003, size=(unit_num, 3)),
        sharp_surface[unit_num * 2:unit_num * 3] + np.random.normal(scale=0.06, size=(unit_num, 3)),
        sharp_surface[unit_num * 3:unit_num * 4] + np.random.normal(scale=0.01, size=(unit_num, 3)),
        sharp_surface[unit_num * 4:unit_num * 5] + np.random.normal(scale=0.02, size=(unit_num, 3)),
        sharp_surface[unit_num * 5:] + np.random.normal(scale=0.04, size=(len(sharp_surface) - 5 * unit_num, 3))
    ], axis=0)

    np.random.shuffle(random_near_points)
    np.random.shuffle(sharp_near_points)

    try:
        sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    except AttributeError:
        sign_type = None

    vol_sdf = call_signed_distance(vol_points, mesh.vertices, mesh.faces, sign_type=sign_type)
    random_near_sdf = call_signed_distance(random_near_points, mesh.vertices, mesh.faces, sign_type=sign_type)
    sharp_near_sdf = call_signed_distance(sharp_near_points, mesh.vertices, mesh.faces, sign_type=sign_type)

    vol_label = -vol_sdf
    random_near_label = -random_near_sdf
    sharp_near_label = -sharp_near_sdf

    data = {
        "vol_points": vol_points.astype(np.float16),
        "vol_label": vol_label.astype(np.float16),
        "random_near_points": random_near_points.astype(np.float16),
        "random_near_label": random_near_label.astype(np.float16),
        "sharp_near_points": sharp_near_points.astype(np.float16),
        "sharp_near_label": sharp_near_label.astype(np.float16),
    }
    return data


def SampleMesh(V: np.ndarray, F: np.ndarray):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    sample_num = 499712 // 4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)

    surface_data = {
        "random_surface": surface,
        "sharp_surface": sharp_surface,
    }

    sdf_data = sample_sdf(mesh, random_surface, random_sharp_surface)
    return surface_data, sdf_data


def normalize_to_unit_box(V: np.ndarray):
    V = np.asarray(V, dtype=np.float64)
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    scale = (V_max - V_min).max() * 1.01
    V_normalized = (V - V_min) / scale
    return V_normalized


# Given: V (n x 3 array of vertices), F (m x 3 array of faces)
# Parameters epsilon/grid_res

def Watertight(V: np.ndarray, F: np.ndarray, epsilon: float = 2.0 / 256, grid_res: int = 256):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float64)

    try:
        sign_type = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    except AttributeError:
        sign_type = None

    sdf = call_signed_distance(grid_points, V, F, sign_type=sign_type)

    ret = igl.marching_cubes(epsilon - np.abs(sdf), grid_points, grid_res, grid_res, grid_res, 0.0)
    mc_verts = ret[0]
    mc_faces = ret[1]
    return mc_verts, mc_faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a mesh file (OBJ/PLY/...) and output surface and SDF data.')
    parser.add_argument('--input_obj', type=str, required=True, help='Path to the input mesh file (OBJ/PLY/...)')
    parser.add_argument('--output_prefix', type=str, required=True, help='Base path for outputs')
    parser.add_argument('--grid_res', type=int, default=256, help='Grid resolution for watertight marching cubes')
    parser.add_argument('--epsilon', type=float, default=2.0/256, help='Isosurface epsilon for marching cubes')
    args = parser.parse_args()

    input_path = args.input_obj
    output_prefix = args.output_prefix

    V, F = read_mesh_vertices_faces(input_path)
    V = normalize_to_unit_box(V)

    mc_verts, mc_faces = Watertight(V, F, epsilon=args.epsilon, grid_res=args.grid_res)
    surface_data, sdf_data = SampleMesh(mc_verts, mc_faces)

    parent_folder = os.path.dirname(output_prefix)
    if len(parent_folder) > 0:
        os.makedirs(parent_folder, exist_ok=True)

    export_surface = f'{output_prefix}_surface.npz'
    np.savez(export_surface, **surface_data)

    export_sdf = f'{output_prefix}_sdf.npz'
    np.savez(export_sdf, **sdf_data)

    out_path = f'{output_prefix}_watertight.obj'
    if hasattr(igl, 'write_obj'):
        igl.write_obj(out_path, mc_verts, mc_faces)
    else:
        igl.writeOBJ(out_path, mc_verts, mc_faces) 