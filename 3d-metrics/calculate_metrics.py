import json
import os
import re
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from metrics_final_runner import MetricsConfig
from metric_utils import compute_cd_and_f_score, compute_IoU

GT = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train-Test-Set/PartCrafter/test/merged-test<16.json'
infer_result_path = {
                     'PartCrafter': '/opt/liblibai-models/user-workspace/colabrate/wenda/results/PartCrafter/merged-test-1027/20251027_22_25_32',
                     'PartCrafter_tuned_12k': '/opt/liblibai-models/user-workspace/colabrate/wenda/results/PartCrafter/converted-test-1031/20251031_003340_finetuned_2_training_step=12000',
                     'PartCrafter_tuned_28k': '/opt/liblibai-models/user-workspace/colabrate/wenda/results/PartCrafter/converted-test-1031/20251031_003340_finetuned_2_training=28000',
                     'PartCrafter_tuned_52k': '/opt/liblibai-models/user-workspace/colabrate/wenda/results/PartCrafter/converted-test-1031/20251031_003340_finetuned_2_-52k_52000'
                     }

# 输出路径配置（基础路径，会根据模型名称动态生成）
OUTPUT_BASE_DIR = '/opt/liblibai-models/user-workspace/colabrate/wenda/eval/DiFa/baselines'
MAX_WORKERS = max(1, min(16, os.cpu_count() or 1))  # 并行进程数，可根据机器CPU调整
MASK_RENDER_DIR = os.environ.get('MASK_RENDER_DIR')
METRICS_F_SCORE_THRESHOLD = float(os.environ.get('METRICS_F_SCORE_THRESHOLD', 0.03))
METRICS_CD_SAMPLES = int(os.environ.get('METRICS_CD_SAMPLES', 100000))
METRICS_CONFIG = MetricsConfig(
    f_score_threshold=METRICS_F_SCORE_THRESHOLD,
    cd_num_samples=METRICS_CD_SAMPLES,
    iou_render_dir=MASK_RENDER_DIR
)

def load_jsonl(path):
    """加载 jsonl 列表"""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_jsonl(path, record):
    """向 jsonl 文件追加一条记录"""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_progress(progress_path, data):
    """保存进度信息"""
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

output_format = """
# Dataset | DiFa-lowpoly | DiFa-highpoly | Cloth4D |
# --------|-------------|-------------|--------|
# Metrics | CD(Chamfer Distance) / F-Score / IoU |
# PartCrafter | 1/2/3 | 1/2/3 | 1/2/3 |
# PartCrafter_tuned_12k | 1/2/3 | 1/2/3 | 1/2/3 |
# PartCrafter_tuned_28k | 1/2/3 | 1/2/3 | 1/2/3 |
# PartCrafter_tuned_52k | 1/2/3 | 1/2/3 | 1/2/3 |"

# We evaluate the fidelity of generated 3D meshes by L2 Chamfer Distance (CD) and F-Score with a threshold of 0.1
# Geometry Independence of Generated Part Meshes. We use the Average Intersection over Union (IoU) to evaluate the geometry independence of generated part meshes. We compute the average IoU between each generated part by voxelizing the canonical space into 64x64x64 grids. Lower IoU indicates less overlap between generated parts, thus demonstrating better part independence.   
"""


def extract_filename_from_mesh_path(mesh_path):
    """从mesh_path提取文件名，并转换为infer路径格式"""
    # 例如: HighPoly_2667_thin.glb -> HighPoly_02667_thin_parts_8
    #      LowPoly_0707.glb -> LowPoly_0007_parts_2
    filename = os.path.basename(mesh_path)  # HighPoly_2667_thin.glb
    name_without_ext = os.path.splitext(filename)[0]  # HighPoly_2667_thin
    
    # 提取数字部分
    match = re.search(r'(\d+)', name_without_ext)
    if match:
        num = int(match.group(1))
        # 根据前缀决定补0位数：LowPoly、z系列用4位，HighPoly等用5位
        prefix = name_without_ext
        if prefix.startswith('LowPoly') or prefix.startswith('z'):
            num_str = f"{num:04d}"
        else:
            num_str = f"{num:05d}"
        # 替换原数字为补0后的数字
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
        print(f"Warning: {glb_path} does not exist")
        return None
    
    try:
        # trimesh可以加载GLB文件
        mesh = trimesh.load(glb_path)
        
        # 如果是Scene对象，合并所有mesh
        if isinstance(mesh, trimesh.Scene):
            # 获取所有几何体
            geometries = list(mesh.geometry.values())
            if len(geometries) == 1:
                mesh = geometries[0]
            else:
                # 合并多个mesh
                mesh = trimesh.util.concatenate(geometries)
        
        return mesh
    except Exception as e:
        print(f"Error loading {glb_path}: {e}")
        return None


def align_meshes(mesh1, mesh2, center_align=True, scale_align=False):
    """对齐两个mesh到相同的位置和尺寸
    
    Args:
        mesh1: 第一个mesh（通常是GT）
        mesh2: 第二个mesh（通常是infer）
        center_align: 是否中心对齐（将两个mesh都移到原点）
        scale_align: 是否尺寸对齐（将两个mesh归一化到相同尺寸）
    
    Returns:
        aligned_mesh1, aligned_mesh2: 对齐后的mesh副本
    """
    mesh1_aligned = mesh1.copy()
    mesh2_aligned = mesh2.copy()
    
    if center_align:
        # 计算中心点
        center1 = (mesh1_aligned.bounds[0] + mesh1_aligned.bounds[1]) / 2
        center2 = (mesh2_aligned.bounds[0] + mesh2_aligned.bounds[1]) / 2
        
        # 移动到原点
        mesh1_aligned.apply_translation(-center1)
        mesh2_aligned.apply_translation(-center2)
    
    if scale_align:
        # 计算尺寸（使用最大维度）
        size1 = np.max(mesh1_aligned.bounds[1] - mesh1_aligned.bounds[0])
        size2 = np.max(mesh2_aligned.bounds[1] - mesh2_aligned.bounds[0])
        
        # 归一化到单位尺寸（使用mesh1的尺寸作为参考）
        if size1 > 1e-6:
            scale1 = 1.0 / size1
            mesh1_aligned.apply_scale(scale1)
        if size2 > 1e-6:
            scale2 = 1.0 / size2
            mesh2_aligned.apply_scale(scale2)
    
    return mesh1_aligned, mesh2_aligned


def calculate_mesh_metrics(
    gt_mesh: trimesh.Trimesh,
    infer_mesh: trimesh.Trimesh,
    num_samples: int,
    threshold: float,
    iou_num_grids: int = 64,
    iou_scale: float = 2.0
):
    """按照统一配置计算 CD / F-Score / IoU。"""
    cd, f_score = compute_cd_and_f_score(
        gt_mesh,
        infer_mesh,
        num_samples=num_samples,
        threshold=threshold
    )
    iou = compute_IoU(gt_mesh, infer_mesh, num_grids=iou_num_grids, scale=iou_scale)
    return {
        'cd': cd,
        'f_score': f_score,
        'iou': iou
    }

def process_single_pair(gt_mesh_path, infer_mesh_path, visualize=True, save_viz_path=None, 
                        align_center=True, align_scale=False, verbose=True):
    """处理单个GT和infer mesh对
    
    Args:
        gt_mesh_path: GT mesh路径
        infer_mesh_path: Infer mesh路径
        visualize: 是否可视化
        save_viz_path: 可视化保存路径
        align_center: 是否中心对齐（将两个mesh都移到原点）- 推荐True
        align_scale: 是否尺寸对齐（将两个mesh归一化到相同尺寸）- False=评估包含尺寸，True=只评估形状
    """
    if verbose:
        print(f"\nProcessing:")
        print(f"  GT: {gt_mesh_path}")
        print(f"  Infer: {infer_mesh_path}")
    
    # 加载mesh
    gt_mesh = load_glb_mesh(gt_mesh_path)
    infer_mesh = load_glb_mesh(infer_mesh_path)
    
    if gt_mesh is None or infer_mesh is None:
        if verbose:
            print("  Error: Failed to load meshes")
        return None
    
    # 打印原始边界框信息
    if verbose:
        print(f"\n  Original Mesh Info:")
        print(f"    GT bounds: min={gt_mesh.bounds[0]}, max={gt_mesh.bounds[1]}")
        print(f"    Infer bounds: min={infer_mesh.bounds[0]}, max={infer_mesh.bounds[1]}")
    
    # 对齐mesh（如果需要）
    if align_center or align_scale:
        if verbose:
            print(f"\n  Aligning meshes (center={align_center}, scale={align_scale})...")
        gt_mesh_aligned, infer_mesh_aligned = align_meshes(gt_mesh, infer_mesh, 
                                                           center_align=align_center, 
                                                           scale_align=align_scale)
        if verbose:
            print(f"    After alignment:")
            print(f"      GT bounds: min={gt_mesh_aligned.bounds[0]}, max={gt_mesh_aligned.bounds[1]}")
            print(f"      Infer bounds: min={infer_mesh_aligned.bounds[0]}, max={infer_mesh_aligned.bounds[1]}")
        gt_mesh = gt_mesh_aligned
        infer_mesh = infer_mesh_aligned
    
    # 可视化（暂时取消）
    # if visualize:
    #     print(f"\n  Visualizing meshes...")
    #     viz_start = time.time()
    #     visualize_meshes(gt_mesh, infer_mesh, save_path=save_viz_path)
    #     viz_time = time.time() - viz_start
    #     print(f"    Visualization took {viz_time:.2f} seconds")
    
    # 计算指标（使用对齐后的mesh）
    if verbose:
        print(f"\n  Computing metrics...")
        print(f"    GT mesh: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
        print(f"    Infer mesh: {len(infer_mesh.vertices)} vertices, {len(infer_mesh.faces)} faces")

    metrics = calculate_mesh_metrics(
        gt_mesh,
        infer_mesh,
        num_samples=METRICS_CONFIG.cd_num_samples,
        threshold=METRICS_CONFIG.f_score_threshold
    )

    if verbose:
        print(f"\n  Final Metrics:")
        print(f"    CD: {metrics['cd']:.6f}")
        print(f"    F-Score: {metrics['f_score']:.6f}")
        print(f"    IoU: {metrics['iou']:.6f}")
    
    return metrics


def process_sample_task(args):
    """多进程任务包装"""
    (sample_idx, sample, model_name, infer_base, viz_path) = args
    gt_mesh_path = sample['mesh_path']
    num_parts = sample['num_parts']
    dataset = sample.get('dataset', 'unknown')
    file_name = os.path.basename(gt_mesh_path)
    infer_mesh_path = build_infer_path(gt_mesh_path, num_parts, infer_base)

    try:
        metrics = process_single_pair(
            gt_mesh_path,
            infer_mesh_path,
            visualize=False,
            save_viz_path=viz_path,
            align_center=True,
            align_scale=False,
            verbose=False
        )
        if metrics is None:
            return {
                'status': 'skip',
                'sample_idx': sample_idx,
                'file': file_name,
                'dataset': dataset,
                'reason': 'process_single_pair returned None'
            }

        return {
            'status': 'success',
            'sample_idx': sample_idx,
            'file': file_name,
            'dataset': dataset,
            'num_parts': num_parts,
            'cd': metrics['cd'],
            'f_score': metrics['f_score'],
            'iou': metrics['iou'],
            'viz_path': viz_path
        }
    except Exception as e:
        return {
            'status': 'error',
            'sample_idx': sample_idx,
            'file': file_name,
            'dataset': dataset,
            'reason': str(e)
        }


if __name__ == "__main__":
    # 读取GT JSON文件
    with open(GT, 'r') as f:
        gt_data = json.load(f)

    total_samples = len(gt_data)
    print(f"Loaded {total_samples} samples from GT JSON\n")

    # 遍历所有模型
    for model_name, infer_base in infer_result_path.items():
        print(f"\n{'#'*40}")
        print(f"Evaluating model: {model_name}")
        print(f"{'#'*40}\n")

        # 根据模型名称动态生成输出路径
        OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, model_name)
        VIZ_DIR = os.path.join(OUTPUT_DIR, 'visualizations')  # 可视化图片保存目录
        RESULTS_MD = os.path.join(OUTPUT_DIR, 'metrics_results.md')  # 结果MD文件
        CACHE_RESULTS = os.path.join(OUTPUT_DIR, 'results_cache.jsonl')
        CACHE_ERRORS = os.path.join(OUTPUT_DIR, 'errors_cache.jsonl')
        PROGRESS_LOG = os.path.join(OUTPUT_DIR, 'progress_log.json')

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(VIZ_DIR, exist_ok=True)

        cached_results = load_jsonl(CACHE_RESULTS)
        cached_errors = load_jsonl(CACHE_ERRORS)
        processed_set = {item['sample_idx'] for item in cached_results}

        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Results will be saved to: {RESULTS_MD}")
        print(f"Using up to {MAX_WORKERS} workers")
        print(f"Cached successes: {len(cached_results)}, cached errors: {len(cached_errors)}\n")

        tasks = []
        for i, sample in enumerate(gt_data):
            if i in processed_set:
                continue
            viz_filename = f"{model_name}_{i:04d}_{os.path.splitext(os.path.basename(sample['mesh_path']))[0]}.png"
            viz_path = os.path.join(VIZ_DIR, viz_filename)
            tasks.append((i, sample, model_name, infer_base, viz_path))

        all_results = cached_results[:]
        errors = cached_errors[:]
        processed_counter = len(all_results) + len(errors)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(process_sample_task, task)
                future_to_task[future] = task

            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Samples ({model_name})"):
                task = future_to_task[future]
                idx = task[0]
                file_name = os.path.basename(task[1]['mesh_path'])
                try:
                    result = future.result()
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    errors.append({
                        'status': 'error',
                        'sample_idx': idx,
                        'file': file_name,
                        'dataset': task[1].get('dataset', 'unknown'),
                        'reason': err_msg
                    })
                    print(f"  Error: sample {idx} ({file_name}) crashed ({err_msg})")
                    continue

                if result['status'] == 'success':
                    all_results.append(result)
                    append_jsonl(CACHE_RESULTS, result)
                else:
                    errors.append(result)
                    append_jsonl(CACHE_ERRORS, result)
                    print(f"  Warning: sample {result['sample_idx']} ({result['file']}) skipped ({result['reason']})")

                processed_counter += 1
                if processed_counter % 10 == 0:
                    save_progress(PROGRESS_LOG, {
                        'model': model_name,
                        'processed': len(all_results),
                        'errors': len(errors),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })

        # 生成MD结果文件
        print(f"\n{'='*80}")
        print(f"Generating results summary for {model_name}...")
        print(f"{'='*80}")

        all_results_sorted = sorted(all_results, key=lambda x: x['sample_idx'])

        with open(RESULTS_MD, 'w', encoding='utf-8') as f:
            f.write("# 3D Metrics Evaluation Results\n\n")
            f.write(f"**Model**: {model_name}\n\n")
            f.write(f"**Total Samples Processed**: {len(all_results_sorted)} / {total_samples}\n\n")
            f.write(f"**Failed Samples**: {len(errors)}\n\n")
            if errors:
                f.write("## Failed Samples\n\n")
                for err in errors[:20]:
                    f.write(f"- Sample {err['sample_idx']+1} ({err['file']}): {err['reason']}\n")
                f.write("\n")

            f.write("## Metrics Explanation\n\n")
            f.write("- **CD (Chamfer Distance)**: L2 Chamfer Distance, lower is better\n")
            f.write("- **F-Score**: F-Score with threshold 0.1, higher is better (0-1)\n")
            f.write("- **IoU**: Intersection over Union, higher is better (0-1)\n\n")
            f.write("## Detailed Results\n\n")
            f.write("| Sample | File | Dataset | Parts | CD | F-Score | IoU | Visualization |\n")
            f.write("|--------|------|---------|-------|----|---------|-----|---------------|\n")

            for result in all_results_sorted:
                viz_rel_path = os.path.relpath(result['viz_path'], OUTPUT_DIR)
                f.write(f"| {result['sample_idx']+1} | {result['file']} | {result['dataset']} | "
                       f"{result['num_parts']} | {result['cd']:.6f} | {result['f_score']:.6f} | "
                       f"{result['iou']:.6f} | [View]({viz_rel_path}) |\n")

            # 计算平均值
            if len(all_results_sorted) > 0:
                avg_cd = np.mean([r['cd'] for r in all_results_sorted])
                avg_f_score = np.mean([r['f_score'] for r in all_results_sorted])
                avg_iou = np.mean([r['iou'] for r in all_results_sorted])

                f.write("\n## Average Metrics\n\n")
                f.write("| Metric | Average |\n")
                f.write("|--------|----------|\n")
                f.write(f"| CD | {avg_cd:.6f} |\n")
                f.write(f"| F-Score | {avg_f_score:.6f} |\n")
                f.write(f"| IoU | {avg_iou:.6f} |\n")

        print(f"\nResults saved to: {RESULTS_MD}")
        if len(all_results_sorted) > 0:
            avg_cd = np.mean([r['cd'] for r in all_results_sorted])
            avg_f_score = np.mean([r['f_score'] for r in all_results_sorted])
            avg_iou = np.mean([r['iou'] for r in all_results_sorted])
            print(f"Summary for {model_name}:")
            print(f"  Average CD: {avg_cd:.6f}")
            print(f"  Average F-Score: {avg_f_score:.6f}")
            print(f"  Average IoU: {avg_iou:.6f}")
        else:
            print("  No successful samples processed.")

        if errors:
            print(f"  Failed samples: {len(errors)} (details in report)")

        save_progress(PROGRESS_LOG, {
            'model': model_name,
            'processed': len(all_results_sorted),
            'errors': len(errors),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })