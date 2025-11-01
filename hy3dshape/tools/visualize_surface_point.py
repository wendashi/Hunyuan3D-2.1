import os
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # 用于服务器无界面绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 高对比调色板
HIGH_CONTRAST_COLORS = [
    (0.121, 0.466, 0.705),  # blue
    (1.000, 0.498, 0.054),  # orange
]

def set_axes_equal(ax):
    """使 3D 坐标轴等比例显示，避免失真。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def maybe_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]

def visualize_surface_npz(npz_path: str, save_path: str = None, show: bool = False, max_points: int = 200000, seed: int = 42):
    """
    可视化 _surface.npz 文件，包含 random_surface / sharp_surface 点集（形状 N,6，前三列为坐标，后三列为法线）。
    """
    random.seed(seed)
    np.random.seed(seed)
    assert os.path.exists(npz_path), f"文件不存在: {npz_path}"
    data = np.load(npz_path)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    legends = []

    if 'random_surface' in data:
        pts = data['random_surface']
        pts = maybe_subsample(pts, max_points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=[HIGH_CONTRAST_COLORS[0]], alpha=0.8, label="random_surface")
        legends.append('random_surface')
    if 'sharp_surface' in data:
        pts = data['sharp_surface']
        pts = maybe_subsample(pts, max_points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=[HIGH_CONTRAST_COLORS[1]], alpha=0.8, label="sharp_surface")
        legends.append('sharp_surface')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if legends:
        ax.legend(loc='upper right', markerscale=6, fontsize=10)
    set_axes_equal(ax)
    fig.tight_layout()
    # 保存或展示
    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(save_path, dpi=300)
        print(f"已保存: {save_path}")
    if show:
        plt.show()

def main():
    # 用例：可修改为你要可视化的数据路径
    surface_npz = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan/HighPoly_0411_thin/geo_data/HighPoly_0411_thin_surface.npz"
    save_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan/HighPoly_0411_thin/geo_data/visualize_surface.png"
    visualize_surface_npz(surface_npz, save_path, show=False, max_points=80000, seed=42)

if __name__ == '__main__':
    main()