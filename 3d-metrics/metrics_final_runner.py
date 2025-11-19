# metrics_final_runner.py

import os
import trimesh
import numpy as np
import pandas as pd
from PIL import Image # 用于读取 PNG Mask
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 导入您上传的工具函数（用于 CD 和 F-Score）
# 确保 metric_utils.py 在同一目录中
from metric_utils import compute_cd_and_f_score 

# --- 1. 配置路径和参数 ---
BASE_DIR = os.path.expanduser("~/3d_metrics_project") 
DATA_DIR = os.path.join(BASE_DIR, "data_preprocessed")

GT_MESH_DIR = os.path.join(DATA_DIR, "GT_NORM")
PRED_MESH_DIR = os.path.join(DATA_DIR, "PRED_NORM")
IOU_RENDER_DIR = os.path.join(DATA_DIR, "IOU_RENDERS") # Blender 渲染图的路径
RESULT_FILE = os.path.join(BASE_DIR, "final_metrics_report.csv")

# --- 2. 指标参数设置 (基于我们的分析) ---
# (单位：米，因为我们的 .obj 模型是以米为单位的)
F_SCORE_THRESHOLD = 0.03  # 阈值：3 厘米 (0.03 米)
CD_F_SAMPLES = 100000      # 采样点数：100K (用于 CD/F-Score)

# Blender 渲染的视点列表（必须与 Blender 脚本中的 STANDARD_VIEWS 匹配）
STANDARD_VIEWS = [
    (0, 0),        # 正面
    (90, 0),       # 右侧
    (180, 0),      # 背面
    (-90, 0),      # 左侧
]
VIEW_NAMES = [f"AZ{az}_EL{el}" for az, el in STANDARD_VIEWS]


@dataclass
class MetricsConfig:
    """
    控制 CD/F-Score/Mask IoU 计算的参数。
    """
    f_score_threshold: float = F_SCORE_THRESHOLD
    cd_num_samples: int = CD_F_SAMPLES
    standard_views: List[Tuple[int, int]] = field(default_factory=lambda: STANDARD_VIEWS.copy())
    iou_render_dir: Optional[str] = IOU_RENDER_DIR

    @property
    def view_names(self) -> List[str]:
        return [f"AZ{az}_EL{el}" for az, el in self.standard_views]


def compute_mask_iou_from_renders(
    model_base_name: str,
    render_dir: Optional[str],
    view_names: List[str],
) -> float:
    """
    从 Blender 渲染的 PNG 文件计算平均 Mask IoU。
    """
    if not render_dir:
        return np.nan

    iou_scores: List[float] = []

    for view_name in view_names:
        try:
            gt_path = os.path.join(render_dir, f"{model_base_name}_GT_{view_name}.png")
            pred_path = os.path.join(render_dir, f"{model_base_name}_PRED_{view_name}.png")

            # 读取图像并提取 Alpha 通道（Mask）
            # 我们的 Blender 脚本将模型渲染为白色(RGB)，背景为透明(Alpha=0)
            # 因此，Alpha 通道 > 0 就是我们的 Mask
            gt_alpha = np.array(Image.open(gt_path).getchannel('A'))
            pred_alpha = np.array(Image.open(pred_path).getchannel('A'))

            mask_gt = (gt_alpha > 0)
            mask_pred = (pred_alpha > 0)

            # 计算 IoU
            intersection = np.logical_and(mask_gt, mask_pred).sum()
            union = np.logical_or(mask_gt, mask_pred).sum()
            
            if union == 0:
                # 如果 GT 和 PRED 都是空的（罕见），IoU 为 1.0 (完美匹配)
                # 如果一个是空的，一个是有的，IoU 为 0.0
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            
            iou_scores.append(iou)

        except FileNotFoundError:
            print(f"  Warning: Mask file not found for {model_base_name} view {view_name}. Skipping view.")
            continue
        except Exception as e:
            print(f"  Error processing mask IoU for {model_base_name}: {e}")
            continue

    if not iou_scores:
        return np.nan # 如果所有视图都失败了
        
    return np.mean(iou_scores)


def evaluate_mesh_pair(
    mesh_gt: trimesh.Trimesh,
    mesh_pred: trimesh.Trimesh,
    model_base_name: str,
    config: Optional[MetricsConfig] = None,
) -> Dict[str, float]:
    """
    给定已加载的 GT / Pred mesh，计算 Root CD / F-Score / Mask IoU。
    """
    if config is None:
        config = MetricsConfig()

    if not isinstance(mesh_pred, trimesh.Trimesh) or not isinstance(mesh_gt, trimesh.Trimesh):
        raise ValueError("mesh_gt 和 mesh_pred 必须是 trimesh.Trimesh 对象")

    cd_m2, f_score = compute_cd_and_f_score(
        mesh_gt,
        mesh_pred,
        num_samples=config.cd_num_samples,
        threshold=config.f_score_threshold
    )
    root_cd_cm = np.sqrt(cd_m2) * 100

    mask_iou = compute_mask_iou_from_renders(
        model_base_name,
        config.iou_render_dir,
        config.view_names
    )

    return {
        'cd_m2': cd_m2,
        'root_cd_cm': root_cd_cm,
        'f_score': f_score,
        'mask_iou': mask_iou
    }


def evaluate_pair(
    gt_mesh_path: str,
    pred_mesh_path: str,
    model_base_name: str,
    config: Optional[MetricsConfig] = None,
) -> Dict[str, float]:
    """
    从路径加载 mesh 后再计算指标。
    """
    mesh_pred = trimesh.load_mesh(pred_mesh_path)
    mesh_gt = trimesh.load_mesh(gt_mesh_path)
    return evaluate_mesh_pair(mesh_gt, mesh_pred, model_base_name, config=config)


def run_final_evaluation():
    """执行批量指标计算"""
    print("--- Starting Final Metric Evaluation (CD, F-Score, Mask IoU) ---")
    
    results: List[dict] = []
    config = MetricsConfig()
    
    # 遍历 PRED 目录下的所有标准化 OBJ 文件
    model_files = [f for f in os.listdir(PRED_MESH_DIR) if f.endswith('_m_norm.obj')]

    if not model_files:
        print(f"Error: No prediction .obj files found in {PRED_MESH_DIR}.")
        return

    print(f"Found {len(model_files)} models for evaluation.")

    for filename in model_files:
        model_base_name = filename.replace('_pred_m_norm.obj', '')
        
        pred_path = os.path.join(PRED_MESH_DIR, filename)
        gt_path = os.path.join(GT_MESH_DIR, f"{model_base_name}_gt_m_norm.obj") 

        if not os.path.exists(gt_path):
            print(f"Skipping {model_base_name}: Corresponding GT file not found.")
            continue
            
        print(f"Processing {model_base_name}...")

        try:
            metrics = evaluate_pair(
                gt_mesh_path=gt_path,
                pred_mesh_path=pred_path,
                model_base_name=model_base_name,
                config=config
            )

            # 5. 记录结果
            results.append({
                'ModelName': model_base_name,
                'RootCD_cm': metrics['root_cd_cm'], # 报告 (cm)
                'F_Score': metrics['f_score'],        # 报告 (0-1)
                'Mask_IoU': metrics['mask_iou']       # 报告 (0-1)
            })
            print(f"  Result: RootCD={metrics['root_cd_cm']:.2f} cm, F-Score={metrics['f_score']:.4f}, MaskIoU={metrics['mask_iou']:.4f}")

        except Exception as e:
            print(f"  FATAL Error occurred for {model_base_name}: {e}")
            results.append({'ModelName': model_base_name, 'RootCD_cm': np.nan, 'F_Score': np.nan, 'Mask_IoU': np.nan})

    # 6. 结果输出
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # 计算总体平均值
        mean_metrics = results_df[['RootCD_cm', 'F_Score', 'Mask_IoU']].mean()
        mean_metrics['ModelName'] = 'AVERAGE'
        results_df.loc[len(results_df)] = mean_metrics
        
        print("\n--- Summary Statistics (AVERAGE) ---")
        print(results_df.loc[results_df['ModelName'] == 'AVERAGE'])
        
        # 保存结果
        results_df.to_csv(RESULT_FILE, index=False)
        print(f"\nFinal report saved to {RESULT_FILE}")
    else:
        print("\nNo models were successfully processed.")

if __name__ == "__main__":
    run_final_evaluation()