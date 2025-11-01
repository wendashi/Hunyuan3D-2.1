# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
import io
import sys
import time
import random
import traceback
from typing import Optional, Union, List, Tuple, Dict

import json
import glob
import cv2
import numpy as np
import trimesh

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info

from .utils import worker_init_fn, pytorch_worker_seed, make_seed


class ResampledShards(torch.utils.data.dataset.IterableDataset):
    def __init__(self, datalist, nshards=sys.maxsize, worker_seed=None, deterministic=False):
        super().__init__()
        self.datalist = datalist
        self.nshards = nshards
        # If no worker_seed provided, use pytorch_worker_seed function; else use given seed
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        self.epoch += 1
        if self.deterministic:
            seed = make_seed(self.worker_seed(), self.epoch)
        else:
            seed = make_seed(self.worker_seed(), self.epoch, 
                             os.getpid(), time.time_ns(), os.urandom(4))
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.datalist) - 1)
            yield self.datalist[index]

            
def read_npz(data):
    # Load a numpy .npz file from a file path or file-like object
    # The commented line shows how to load from bytes in memory
    # return np.load(io.BytesIO(data))
    return np.load(data)


def read_json(path):
    # Read and parse a JSON file from the given file path
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# ==================== 坐标转换函数 ====================
# 将点云从 Z-up 坐标系 (x,y ∈ [-1,1], z ∈ [0,2]) 转换为 Y-up 坐标系 (x,y,z ∈ [0,1])
# 基于 to_Hunyuan_torch.py 的实现

# 旋转矩阵：绕 X 轴 -90°，使得 [x, y, z] @ R = [x, z, -y]
# 从 Z-up 转换到 Y-up
ROTATION_Z_UP_TO_Y_UP = torch.tensor([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=torch.float64)


def rotate_z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    """
    将点云从 Z-up 坐标系旋转到 Y-up 坐标系（不做缩放/平移）。
    
    参数:
        points: (N, 3) numpy 数组，输入坐标为 Z-up 坐标系
               x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 2]
    
    返回:
        (N, 3) numpy 数组，输出坐标为 Y-up 坐标系
        旋转后：x' = x, y' = z, z' = -y
        范围：x' ∈ [-1, 1], y' ∈ [0, 2], z' ∈ [1, -1] → [-1, 1]
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points 必须是形状为 (N, 3) 的数组')
    
    orig_dtype = points.dtype
    t = torch.from_numpy(points.astype(np.float64, copy=False))
    # 右乘旋转矩阵
    out = (t @ ROTATION_Z_UP_TO_Y_UP).numpy()
    return out.astype(orig_dtype, copy=False)


def scale_to_target_ranges(points_y_up: np.ndarray) -> np.ndarray:
    """
    在完成 Z-up→Y-up 旋转后，对坐标分别缩放到目标范围 [0, 1]：
    - x: [-1, 1] → [0, 1]  via (x + 1) / 2
    - y:  [0, 2] → [0, 1]  via  y / 2
    - z: [-1, 1] → [0, 1]  via (z + 1) / 2
    
    参数:
        points_y_up: (N, 3) numpy 数组，Y-up 坐标系下的点
                     x ∈ [-1, 1], y ∈ [0, 2], z ∈ [-1, 1]
    
    返回:
        (N, 3) numpy 数组，所有坐标都在 [0, 1] 范围内
    """
    if points_y_up.ndim != 2 or points_y_up.shape[1] != 3:
        raise ValueError('points_y_up 必须是形状为 (N, 3) 的数组')
    
    orig_dtype = points_y_up.dtype
    t = torch.from_numpy(points_y_up.astype(np.float64, copy=False))
    # x in [-1,1] -> [0,1]
    t[:, 0] = (t[:, 0] + 1.0) / 2.0
    # y in [0,2] -> [0,1]
    t[:, 1] = t[:, 1] / 2.0
    # z in [-1,1] -> [0,1]
    t[:, 2] = (t[:, 2] + 1.0) / 2.0
    return t.numpy().astype(orig_dtype, copy=False)


def transform_points_z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    """
    将点云从 Z-up 坐标系转换为 Y-up 坐标系并缩放到 [0, 1]。
    
    完整转换流程：
    1. 旋转：Z-up → Y-up (x, y, z) → (x, z, -y)
    2. 缩放：各轴缩放到 [0, 1]
    
    参数:
        points: (N, 3) numpy 数组，输入坐标为 Z-up 坐标系
               x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 2]
    
    返回:
        (N, 3) numpy 数组，输出坐标为 Y-up 坐标系，所有值在 [0, 1] 范围内
    """
    # 先旋转到 Y-up
    points_y_up = rotate_z_up_to_y_up(points)
    # 再缩放到 [0, 1]
    return scale_to_target_ranges(points_y_up)


def transform_normals_z_up_to_y_up(normals: np.ndarray) -> np.ndarray:
    """
    将法线向量从 Z-up 坐标系旋转到 Y-up 坐标系（仅旋转，不缩放）。
    
    参数:
        normals: (N, 3) numpy 数组，输入法线为 Z-up 坐标系
    
    返回:
        (N, 3) numpy 数组，输出法线为 Y-up 坐标系（已归一化）
    """
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError('normals 必须是形状为 (N, 3) 的数组')
    
    orig_dtype = normals.dtype
    t = torch.from_numpy(normals.astype(np.float64, copy=False))
    # 旋转
    rot = t @ ROTATION_Z_UP_TO_Y_UP
    # 归一化以避免数值误差
    rot = torch.nn.functional.normalize(rot, dim=1, eps=1e-12)
    return rot.numpy().astype(orig_dtype, copy=False)
# ==================== 坐标转换函数结束 ====================


def padding(image, mask, center=True, padding_ratio_range=[1.15, 1.15]):
    """
    Pad the input image and mask to a square shape with padding ratio.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        mask (np.ndarray): Corresponding mask array of shape (H, W).
        center (bool): Whether to center the original image in the padded output.
        padding_ratio_range (list): Range [min, max] to randomly select padding ratio.

    Returns:
        newimg (np.ndarray): Padded image of shape (resize_side, resize_side, 3).
        newmask (np.ndarray): Padded mask of shape (resize_side, resize_side).
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Select padding ratio either fixed or randomly within the given range
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)
    # resize_side = int(max_side * 1.15)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
        
    start_w = pad_w // 2

    # Create new white image and black mask with padded size
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    
    # Place original image and mask into the padded canvas
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    
    return newimg, newmask


def viz_pc(surface, normal, image_input, name):
    image_input = image_input.cpu().numpy()
    image_input = image_input.transpose(1, 2, 0) * 0.5 + 0.5
    image_input = (image_input * 255).astype(np.uint8)
    cv2.imwrite(name + '.png', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
    surface = surface.cpu().numpy()
    normal = normal.cpu().numpy()
    surface_mesh = trimesh.Trimesh(surface, vertex_colors=(normal + 1) / 2)
    surface_mesh.export(name + '.obj')


class AlignedShapeLatentDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(
        self,
        data_list: str = None,
        cond_stage_key: str = "image",
        image_transform = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        deterministic = False,
        worker_seed = None,
        padding = True,
        padding_ratio_range=[1.15, 1.15]
    ):
        super().__init__()
        if isinstance(data_list, str) and data_list.endswith('.json'):
            self.data_list = read_json(data_list)
        elif isinstance(data_list, str) and os.path.isdir(data_list):
            self.data_list = glob.glob(data_list + '/*')
        else:
            self.data_list = data_list
        assert isinstance(self.data_list, list)
        self.rng = random.Random(0)
        
        self.cond_stage_key = cond_stage_key
        self.image_transform = image_transform
        
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Dataset Infos (Ours - Multi-Part Support):')
        rank_zero_info(f'# of 3D file: {len(self.data_list)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'Multi-part support: Enabled')
        rank_zero_info(f'*' * 50)


    def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
        surface_normal = []
        if self.pc_size > 0:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)
            
        if self.pc_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)
            
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size
        
        geo_points = 0.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface, geo_points

    def load_render(self, imgs_path):
        imgs_choice = self.rng.sample(imgs_path, 1)
        images, masks = [], []
        for image_path in imgs_choice:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            forground = image[:, :, :3]
            background = np.ones_like(forground) * 255
            img_new = forground * alpha + background * (1 - alpha)
            image = img_new.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (alpha[:, :, 0] * 255).astype(np.uint8)


            if self.padding:
                h, w = image.shape[:2]
                binary = mask > 0.3
                non_zero_coords = np.argwhere(binary)
                if len(non_zero_coords) > 0:
                    x_min, y_min = non_zero_coords.min(axis=0)
                    x_max, y_max = non_zero_coords.max(axis=0)
                    image, mask = padding(
                        image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        center=True, padding_ratio_range=self.padding_ratio_range)
                else:
                    # 如果 mask 全为0，直接进行 padding（不裁剪）
                    image, mask = padding(image, mask, center=True, padding_ratio_range=self.padding_ratio_range)
            
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
                
            images.append(image)
            masks.append(mask)
            
        # 如果没有成功加载任何图像，返回默认图像
        if len(images) == 0:
            default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
            default_mask = np.zeros((518, 518), dtype=np.uint8)
            if self.image_transform:
                default_image = self.image_transform(default_image)
                default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                default_mask = self.image_transform(default_mask)
            return default_image, default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
        
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)[:1, ...]
        return images, masks

    def decode(self, item):
        uid = item.split('/')[-1]
        render_img_paths = [os.path.join(item, f'render_cond/{i:03d}.png') for i in range(24)]
        # transforms_json_path = os.path.join(item, 'render_cond/transforms.json')
        
        # 检查是否存在单个 surface 文件（原始 Hunyuan3D 格式）
        single_surface_path = os.path.join(item, f'geo_data/{uid}_surface.npz')
        if os.path.exists(single_surface_path):
            # 原始 Hunyuan3D 格式：单个 surface 文件
            surface_data = read_npz(single_surface_path)
            sample = {}
            sample["image"] = render_img_paths
            sample["random_surface"] = surface_data['random_surface']
            sample["sharpedge_surface"] = surface_data['sharp_surface']
            return sample
        else:
            # 我们的多 part 格式：查找所有 part 文件
            geo_data_dir = os.path.join(item, 'geo_data')
            part_files = []
            
            # 查找所有 part 文件
            for filename in os.listdir(geo_data_dir):
                if filename.endswith('_surface.npz') and '_part' in filename:
                    part_files.append(os.path.join(geo_data_dir, filename))
            
            if not part_files:
                raise FileNotFoundError(f"No surface files found in {geo_data_dir}")
            
            # 按 part 编号排序
            part_files.sort()
            
            # 随机选择一个 part 进行训练
            selected_part = self.rng.choice(part_files)
            surface_data = read_npz(selected_part)
            
            sample = {}
            sample["image"] = render_img_paths
            sample["random_surface"] = surface_data['random_surface']
            sample["sharpedge_surface"] = surface_data['sharp_surface']
            sample["selected_part"] = os.path.basename(selected_part)  # 记录选择的 part
            return sample

    def transform(self, sample):
        rng = np.random.default_rng()
        random_surface = sample.get("random_surface", 0)
        sharpedge_surface = sample.get("sharpedge_surface", 0)
        image_input, mask_input = self.load_render(sample['image'])
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
        sample = {
            "surface": surface,
            "geo_points": geo_points,
            "image": image_input,
            "mask": mask_input,
        }
        return sample

    def __iter__(self):
        total_num = 0
        failed_num = 0
        for data in ResampledShards(self.data_list):
            total_num += 1
            if total_num % 1000 == 0:
                print(f"Current failure rate of data loading:")
                print(f"{failed_num}/{total_num}={failed_num/total_num}")
            try:
                sample = self.decode(data)
                sample = self.transform(sample)
            except Exception as err:
                print(err)
                failed_num += 1
                continue
            yield sample


class MultiPartAlignedShapeLatentDataset(torch.utils.data.Dataset):
    """
    支持多 part 的数据集，每个样本返回该物体的所有 parts
    模仿 PartCrafter 的 BatchedObjaversePartDataset 逻辑
    """
    def __init__(
        self,
        data_list: str = None,
        cond_stage_key: str = "image",
        image_transform = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        padding = True,
        padding_ratio_range=[1.15, 1.15],
        batch_size: int = None  # 如果提供，会进行预批次打包
    ):
        super().__init__()
        if isinstance(data_list, str) and data_list.endswith('.json'):
            self.data_list = read_json(data_list)
        elif isinstance(data_list, str) and os.path.isdir(data_list):
            self.data_list = glob.glob(data_list + '/*')
        else:
            self.data_list = data_list
        assert isinstance(self.data_list, list)
        
        self.cond_stage_key = cond_stage_key
        self.image_transform = image_transform
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.batch_size = batch_size
        
        # 解析每个样本的 parts 信息
        self.data_items = []
        for item in self.data_list:
            # 支持两种格式：
            # 1. JSON 格式：字典，包含 surface_path, hunyuan_images_path 等
            # 2. 目录路径格式：字符串路径
            if isinstance(item, dict):
                # JSON 格式（PartCrafter 格式）
                surface_path = item.get('surface_path')
                hunyuan_images_path = item.get('hunyuan_images_path')
                num_parts = item.get('num_parts', 1)
                
                if not surface_path:
                    continue
                    
                self.data_items.append({
                    'surface_path': surface_path,  # .npy 文件路径
                    'images_path': hunyuan_images_path,  # 渲染图像目录
                    'uid': item.get('file', os.path.basename(surface_path).replace('.npy', '')),
                    'num_parts': num_parts,
                    'is_npy_format': True  # 标记是 .npy 格式
                })
            else:
                # 目录路径格式（原始 Hunyuan3D 格式）
                item_path = item
                uid = item_path.split('/')[-1]
                metadata_path = os.path.join(item_path, 'metadata.json')
                num_parts = self._get_num_parts(metadata_path)
                self.data_items.append({
                    'path': item_path,
                    'uid': uid,
                    'num_parts': num_parts,
                    'is_npy_format': False  # 标记是 .npz 格式
                })
        
        # 如果提供了 batch_size，进行预批次打包（类似 BatchedObjaversePartDataset）
        if self.batch_size is not None:
            # 过滤：只保留 num_parts <= batch_size 的数据
            self.data_items = [item for item in self.data_items if item['num_parts'] <= self.batch_size]
            # 预打包成批次
            self.batched_items = self._get_batched_items(self.data_items, self.batch_size)
        else:
            self.batched_items = None
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Multi-Part Dataset Infos:')
        rank_zero_info(f'# of objects: {len(self.data_items)}')
        if self.batched_items is not None:
            rank_zero_info(f'# of batches: {len(self.batched_items)}')
            rank_zero_info(f'Batch size (parts): {self.batch_size}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'*' * 50)
    
    def _get_batched_items(self, data_items, batch_size):
        """
        预打包批次，确保每个批次的 parts 总数刚好等于 batch_size
        模仿 PartCrafter 的 BatchedObjaversePartDataset._get_batched_configs 逻辑
        """
        batched_items = []
        data_items = data_items.copy()  # 避免修改原列表
        
        while len(data_items) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_items = []
            
            # 尝试组合多个物体，使得 parts 总数刚好等于 batch_size
            while temp_num_parts < batch_size and len(data_items) > 0:
                item = data_items.pop()
                num_parts = item['num_parts']
                
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(item)
                    temp_num_parts += num_parts
                else:
                    # 如果加上这个物体会超过 batch_size，放回待处理列表
                    unchosen_items.append(item)
            
            # 将未选中的项放回列表末尾
            data_items = data_items + unchosen_items
            
            if temp_num_parts == batch_size:
                # 成功组成一个批次
                # 如果物体数量少于 batch_size，用空字典填充（类似 PartCrafter）
                if len(temp_batch) < batch_size:
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_items.append(temp_batch)
            else:
                # 无法组成完整的批次（剩余数据不足以组成 batch_size）
                # 丢弃不完整的批次（类似 PartCrafter 的逻辑）
                break
        
        return batched_items
    
    def _get_num_parts(self, metadata_path):
        """从 metadata.json 获取 num_parts"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('num_parts', 1)
        except:
            return 1
    
    def _load_all_parts(self, item_info):
        """加载该物体的所有 parts"""
        parts_data = []
        
        if item_info.get('is_npy_format', False):
            # PartCrafter 格式：从 .npy 文件加载
            surface_path = item_info['surface_path']
            images_path = item_info.get('images_path')
            
            try:
                # 加载 .npy 文件
                data = np.load(surface_path, allow_pickle=True).item()
                
                # 获取 parts 数据
                if 'parts' in data and len(data['parts']) > 0:
                    parts = data['parts']
                else:
                    # 如果没有 parts，使用整个物体
                    if 'object' in data:
                        parts = [data['object']]
                    else:
                        raise ValueError(f"数据文件中既没有 'parts' 也没有 'object': {surface_path}")
                
                # 加载每个 part
                for part in parts:
                    part_data = self._load_single_part_from_dict(part, images_path)
                    if part_data is not None:
                        parts_data.append(part_data)
                        
            except Exception as e:
                print(f"加载 .npy 文件失败 {surface_path}: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            # 原始 Hunyuan3D 格式：从目录中的 .npz 文件加载
            item_path = item_info['path']
            uid = item_info['uid']
            geo_data_dir = os.path.join(item_path, 'geo_data')
            part_files = []
            
            for filename in os.listdir(geo_data_dir):
                if filename.endswith('_surface.npz'):
                    if '_part' in filename:
                        part_files.append(os.path.join(geo_data_dir, filename))
                    elif filename == f'{uid}_surface.npz':
                        # 单文件格式，只有一个 part
                        part_files.append(os.path.join(geo_data_dir, filename))
            
            part_files.sort()
            
            # 加载每个 part
            for part_file in part_files:
                part_data = self._load_single_part(part_file, item_path, uid)
                if part_data is not None:
                    parts_data.append(part_data)
        
        return parts_data
    
    def _load_single_part(self, part_file, item_path, uid):
        """从 .npz 文件加载单个 part 的数据（原始 Hunyuan3D 格式）"""
        try:
            # 加载 surface 数据
            surface_data = read_npz(part_file)
            random_surface = surface_data['random_surface']
            sharp_surface = surface_data['sharp_surface']
            
            # 加载图像数据
            render_img_paths = [os.path.join(item_path, f'render_cond/{i:03d}.png') for i in range(24)]
            image_input, mask_input = self._load_render(render_img_paths)
            
            # 处理 surface 数据
            rng = np.random.default_rng()
            surface, geo_points = self._load_surface_sdf_points(rng, random_surface, sharp_surface)
            
            return {
                'surface': surface,
                'geo_points': geo_points,
                'image': image_input,
                'mask': mask_input,
            }
        except Exception as e:
            print(f"Error loading part {part_file}: {e}")
            return None
    
    def _load_single_part_from_dict(self, part_data, images_path):
        """从字典格式加载单个 part 的数据（PartCrafter .npy 格式），并进行坐标转换"""
        try:
            # part_data 应该是字典格式，包含 surface_points 和 surface_normals
            if isinstance(part_data, dict):
                # 提取点云和法线数据
                if 'surface_points' in part_data and 'surface_normals' in part_data:
                    points = part_data['surface_points']  # [P, 3] Z-up 坐标系
                    normals = part_data['surface_normals']  # [P, 3] Z-up 坐标系
                elif 'points' in part_data and 'normals' in part_data:
                    points = part_data['points']  # [P, 3] Z-up 坐标系
                    normals = part_data['normals']  # [P, 3] Z-up 坐标系
                else:
                    raise ValueError(f"无法找到点云数据，可用键: {list(part_data.keys())}")
                
                # 确保形状正确
                if points.shape[1] != 3 or normals.shape[1] != 3:
                    raise ValueError(f"点云数据形状不正确: points={points.shape}, normals={normals.shape}")
                
                # 坐标转换：从 Z-up 转换为 Y-up
                # 输入：x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 2], Z-up, 中心点 (0, 0, 1)
                # 输出：x, y, z ∈ [0, 1], Y-up, 中心点 (0.5, 0.5, 0.5)
                points_y_up = transform_points_z_up_to_y_up(points)  # [P, 3] Y-up 坐标系，[0, 1]
                normals_y_up = transform_normals_z_up_to_y_up(normals)  # [P, 3] Y-up 坐标系
                
                # 拼接点和法线，格式与 Hunyuan3D 的 random_surface 一致：[P, 6]
                random_surface = np.concatenate([points_y_up, normals_y_up], axis=1)  # [P, 6]
                
                # 如果没有 sharp_surface，使用空数组
                sharp_surface = np.zeros((0, 6), dtype=random_surface.dtype)
            else:
                raise ValueError(f"part_data 必须是字典格式，当前类型: {type(part_data)}")
            
            # 加载图像数据
            try:
                if images_path and os.path.exists(images_path):
                    render_img_paths = [os.path.join(images_path, f'{i:03d}.png') for i in range(24)]
                    # 检查文件是否存在
                    render_img_paths = [p for p in render_img_paths if os.path.exists(p)]
                    if render_img_paths:
                        image_input, mask_input = self._load_render(render_img_paths)
                    else:
                        # 如果图像不存在，创建默认图像（格式与 _load_render 返回一致）
                        default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
                        default_mask = np.zeros((518, 518), dtype=np.uint8)
                        if self.image_transform:
                            default_image = self.image_transform(default_image)
                            default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                            default_mask = self.image_transform(default_mask)
                        image_input = default_image  # [C, H, W]
                        mask_input = default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
                else:
                    # 如果图像路径不存在，创建默认图像（格式与 _load_render 返回一致）
                    default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
                    default_mask = np.zeros((518, 518), dtype=np.uint8)
                    if self.image_transform:
                        default_image = self.image_transform(default_image)
                        default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                        default_mask = self.image_transform(default_mask)
                    image_input = default_image  # [C, H, W]
                    mask_input = default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
            except Exception as img_err:
                print(f"警告：加载图像失败，使用默认图像: {img_err}")
                # 创建默认图像
                default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
                default_mask = np.zeros((518, 518), dtype=np.uint8)
                if self.image_transform:
                    default_image = self.image_transform(default_image)
                    default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                    default_mask = self.image_transform(default_mask)
                image_input = default_image
                mask_input = default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
            
            # 处理 surface 数据（采样和添加标签）
            rng = np.random.default_rng()
            surface, geo_points = self._load_surface_sdf_points(rng, random_surface, sharp_surface)
            
            return {
                'surface': surface,
                'geo_points': geo_points,
                'image': image_input,
                'mask': mask_input,
            }
        except Exception as e:
            print(f"Error loading part from dict: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
        """处理 surface 数据，支持重复采样以处理点数不足的情况"""
        surface_normal = []
        if self.pc_size > 0:
            # 如果点数不够，使用重复采样
            replace = random_surface.shape[0] < self.pc_size
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=replace)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)
            
        if self.pc_sharpedge_size > 0:
            # 如果点数不够，使用重复采样
            replace = sharpedge_surface.shape[0] < self.pc_sharpedge_size
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=replace)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)
            
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size
        
        geo_points = 0.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface, geo_points
    
    def _load_render(self, imgs_path):
        """加载渲染图像，与原始 AlignedShapeLatentDataset 相同"""
        if not imgs_path or len(imgs_path) == 0:
            # 如果没有图像路径，返回默认图像
            default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
            default_mask = np.zeros((518, 518), dtype=np.uint8)
            if self.image_transform:
                default_image = self.image_transform(default_image)
                default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                default_mask = self.image_transform(default_mask)
            return default_image, default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
        
        rng = random.Random(0)
        imgs_choice = rng.sample(imgs_path, min(1, len(imgs_path)))
        images, masks = [], []
        for image_path in imgs_choice:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"警告：无法读取图像 {image_path}，使用默认图像")
                continue
            if len(image.shape) < 3 or image.shape[2] != 4:
                print(f"警告：图像 {image_path} 不是 RGBA 格式，shape={image.shape}，使用默认图像")
                continue
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            forground = image[:, :, :3]
            background = np.ones_like(forground) * 255
            img_new = forground * alpha + background * (1 - alpha)
            image = img_new.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (alpha[:, :, 0] * 255).astype(np.uint8)

            if self.padding:
                h, w = image.shape[:2]
                binary = mask > 0.3
                non_zero_coords = np.argwhere(binary)
                if len(non_zero_coords) > 0:
                    x_min, y_min = non_zero_coords.min(axis=0)
                    x_max, y_max = non_zero_coords.max(axis=0)
                    image, mask = padding(
                        image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        center=True, padding_ratio_range=self.padding_ratio_range)
                else:
                    # 如果 mask 全为0，直接进行 padding（不裁剪）
                    image, mask = padding(image, mask, center=True, padding_ratio_range=self.padding_ratio_range)
            
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
                
            images.append(image)
            masks.append(mask)
            
        # 如果没有成功加载任何图像，返回默认图像
        if len(images) == 0:
            default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
            default_mask = np.zeros((518, 518), dtype=np.uint8)
            if self.image_transform:
                default_image = self.image_transform(default_image)
                default_mask = np.stack((default_mask, default_mask, default_mask), axis=-1)
                default_mask = self.image_transform(default_mask)
            return default_image, default_mask[:1, ...] if isinstance(default_mask, torch.Tensor) else torch.zeros((1, 518, 518))
        
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)[:1, ...]
        return images, masks
    
    def __len__(self):
        if self.batched_items is not None:
            return len(self.batched_items)
        else:
            return len(self.data_items)
    
    def __getitem__(self, idx):
        if self.batched_items is not None:
            # 预批次打包模式：返回一个批次的所有物体
            batch_items = self.batched_items[idx]
            batch_parts_data = []
            
            for item in batch_items:
                if len(item) == 0:
                    # 空占位符，跳过
                    continue
                parts_data = self._load_all_parts(item)
                if parts_data:
                    batch_parts_data.append(parts_data)
            
            # 如果批次为空，返回空列表（会被 collate_fn 过滤）
            return batch_parts_data if batch_parts_data else []
        else:
            # 非批次模式：返回单个物体的所有 parts
            item = self.data_items[idx]
            parts_data = self._load_all_parts(item)
            return parts_data


def collate_multi_part_batch(batch_list, batch_size):
    """
    将多个物体的 parts 组织成固定大小的 batch
    确保每个 batch 正好有 batch_size 个 parts（严格按照 PartCrafter 逻辑）
    
    输入: batch_list = [物体1的parts列表, 物体2的parts列表, ...]
          batch_list 中的每个元素可能为空字典 {}（占位符）
    输出: 一个包含 batch_size 个 parts 的 batch
    
    规则：
    1. parts 总数必须严格等于 batch_size（不能多不能少）
    2. 过滤掉空占位符
    3. num_parts.sum() 必须等于 batch_size
    """
    # 过滤掉空占位符（类似 PartCrafter 的 collate_fn）
    batch_list = [obj_parts for obj_parts in batch_list if len(obj_parts) > 0]
    
    all_parts = []
    num_parts_per_object = []
    
    for obj_parts in batch_list:
        all_parts.extend(obj_parts)
        num_parts_per_object.append(len(obj_parts))
    
    # 验证 parts 总数必须严格等于 batch_size
    total_parts = len(all_parts)
    if total_parts != batch_size:
        raise ValueError(
            f"Parts 总数 ({total_parts}) 必须严格等于 batch_size ({batch_size})。"
            f"当前物体的 num_parts: {num_parts_per_object}, "
            f"总和: {sum(num_parts_per_object)}"
        )
    
    # 打包成 batch
    surfaces = torch.stack([p['surface'] for p in all_parts])
    images = torch.stack([p['image'] for p in all_parts])
    masks = torch.stack([p['mask'] for p in all_parts])
    
    # 验证：按照 PartCrafter 的逻辑
    num_parts_tensor = torch.tensor(num_parts_per_object)
    assert surfaces.shape[0] == images.shape[0] == masks.shape[0] == num_parts_tensor.sum().item() == batch_size, \
        f"批次大小验证失败: surfaces={surfaces.shape[0]}, images={images.shape[0]}, " \
        f"masks={masks.shape[0]}, num_parts.sum()={num_parts_tensor.sum().item()}, batch_size={batch_size}"
    
    return {
        'surface': surfaces,  # [batch_size, num_points, 7]
        'image': images,      # [batch_size, 3, H, W]
        'mask': masks,        # [batch_size, 1, H, W]
        'num_parts': num_parts_tensor  # 记录每个物体有多少 parts，总和必须等于 batch_size
    }


class MultiPartCollateWrapper:
    """
    可 pickle 的 collate_fn wrapper
    用于解决多 GPU 训练时 lambda 函数无法序列化的问题
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def __call__(self, batch):
        # batch 是 DataLoader 返回的，因为 DataLoader batch_size=1，所以取 batch[0]
        return collate_multi_part_batch(batch[0], self.batch_size)


def test_multi_part_dataloader(data_list_path, max_samples=5):
    """
    测试多 part 数据加载器
    
    Args:
        data_list_path: 数据列表路径
        max_samples: 最大测试样本数
    """
    print("=" * 60)
    print("测试多 part 数据加载器")
    print("=" * 60)
    
    # 创建图像变换
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(518),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # 创建多 part 数据集
    dataset = MultiPartAlignedShapeLatentDataset(
        data_list=data_list_path,
        image_transform=image_transform,
        pc_size=81920,  # 匹配 Hunyuan3D 配置
        pc_sharpedge_size=0,  # 匹配 Hunyuan3D 配置
        sharpedge_label=True,
        return_normal=True
    )
    
    print(f"数据集大小: {len(dataset.data_items)}")
    print(f"开始测试前 {max_samples} 个样本...")
    print()
    
    success_count = 0
    for i in range(min(max_samples, len(dataset))):
        try:
            print(f"样本 {i+1}:")
            parts_data = dataset[i]
            print(f"  Parts 数量: {len(parts_data)}")
            
            for j, part_data in enumerate(parts_data):
                print(f"  Part {j}:")
                print(f"    Surface shape: {part_data['surface'].shape}")
                print(f"    Image shape: {part_data['image'].shape}")
                print(f"    Mask shape: {part_data['mask'].shape}")
            
            print(f"  ✅ 加载成功")
            success_count += 1
            print()
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"测试完成: {success_count}/{max_samples} 个样本成功")
    return success_count == max_samples


def test_collate_function(data_list_path, batch_size=4):
    """
    测试批次打包功能
    
    Args:
        data_list_path: 数据列表路径
        batch_size: 批次大小
    """
    print("=" * 60)
    print("测试批次打包功能")
    print("=" * 60)
    
    # 创建图像变换
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(518),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # 创建多 part 数据集
    dataset = MultiPartAlignedShapeLatentDataset(
        data_list=data_list_path,
        image_transform=image_transform,
        pc_size=81920,  # 匹配 Hunyuan3D 配置
        pc_sharpedge_size=0,  # 匹配 Hunyuan3D 配置
        sharpedge_label=True,
        return_normal=True
    )
    
    print(f"数据集大小: {len(dataset.data_items)}")
    print(f"批次大小: {batch_size}")
    print()
    
    # 模拟批次数据
    batch_list = []
    for i in range(min(3, len(dataset))):  # 取前3个物体
        parts_data = dataset[i]
        batch_list.append(parts_data)
        print(f"物体 {i+1} 有 {len(parts_data)} 个 parts")
    
    # 测试批次打包
    try:
        batch = collate_multi_part_batch(batch_list, batch_size)
        print(f"\n批次打包结果:")
        print(f"  Surface shape: {batch['surface'].shape}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Num parts per object: {batch['num_parts']}")
        print(f"  ✅ 批次打包成功")
        return True
    except Exception as e:
        print(f"  ❌ 批次打包失败: {e}")
        import traceback
        traceback.print_exc()
        return False


class BatchedPartCrafterDataset(torch.utils.data.Dataset):
    """
    PartCrafter 风格的数据集，支持预打包批次
    模仿 PartCrafter 的 BatchedObjaversePartDataset 逻辑
    """
    def __init__(
        self,
        data_list: str = None,
        batch_size: int = 4,
        image_transform = None,
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        sharpedge_label: bool = True,
        return_normal: bool = True,
        padding: bool = True,
        padding_ratio_range: List[float] = [1.15, 1.15]
    ):
        super().__init__()
        if isinstance(data_list, str) and data_list.endswith('.json'):
            self.data_configs = read_json(data_list)
        else:
            self.data_configs = data_list
        assert isinstance(self.data_configs, list)
        
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
        # 过滤和预处理数据
        self.data_configs = [c for c in self.data_configs if c.get('valid', True)]
        self.data_configs = [c for c in self.data_configs if c.get('num_parts', 1) <= batch_size]
        
        # 预打包成批次
        self.batched_configs = self._get_batched_configs(self.data_configs, batch_size)
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'PartCrafter Dataset Infos:')
        rank_zero_info(f'# of objects: {len(self.data_configs)}')
        rank_zero_info(f'# of batches: {len(self.batched_configs)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'*' * 50)
    
    def _get_batched_configs(self, data_configs, batch_size):
        """模仿 PartCrafter 的批次打包逻辑"""
        batched = []
        data_configs = data_configs.copy()  # 避免修改原列表
        
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen = []
            
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop()
                if temp_num_parts + config.get('num_parts', 1) <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += config.get('num_parts', 1)
                else:
                    unchosen.append(config)
            
            data_configs = data_configs + unchosen
            
            if temp_num_parts == batch_size:
                batched.append(temp_batch)
            elif len(data_configs) == 0 and temp_num_parts > 0:
                # 如果剩余数据不足一个完整批次，也添加进去
                batched.append(temp_batch)
        
        return batched
    
    def _load_image(self, img_path):
        """加载单张图像"""
        try:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {img_path}")
            
            if image.shape[2] == 4:
                # 处理 RGBA 图像
                alpha = image[:, :, 3:4].astype(np.float32) / 255
                foreground = image[:, :, :3]
                background = np.ones_like(foreground) * 255
                image = foreground * alpha + background * (1 - alpha)
                image = image.astype(np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.padding:
                h, w = image.shape[:2]
                # 简单的中心裁剪和填充
                max_side = max(h, w)
                padding_ratio = self.padding_ratio_range[0] if self.padding_ratio_range[0] == self.padding_ratio_range[1] else np.random.uniform(self.padding_ratio_range[0], self.padding_ratio_range[1])
                resize_side = int(max_side * padding_ratio)
                
                pad_h = resize_side - h
                pad_w = resize_side - w
                start_h = pad_h // 2
                start_w = pad_w // 2
                
                newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
                newimg[start_h:start_h + h, start_w:start_w + w] = image
                image = newimg
            
            if self.image_transform:
                image = self.image_transform(image)
            
            return image
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回一个默认图像
            default_image = np.ones((518, 518, 3), dtype=np.uint8) * 255
            if self.image_transform:
                default_image = self.image_transform(default_image)
            return default_image
    
    def _load_part_surface(self, part_data):
        """加载单个 part 的表面数据，并进行坐标转换"""
        try:
            # part_data 应该是字典格式，包含 surface_points 和 surface_normals
            if isinstance(part_data, dict):
                # 提取点云和法线数据
                if 'surface_points' in part_data and 'surface_normals' in part_data:
                    points = part_data['surface_points']  # [P, 3] Z-up 坐标系
                    normals = part_data['surface_normals']  # [P, 3] Z-up 坐标系
                elif 'points' in part_data and 'normals' in part_data:
                    points = part_data['points']  # [P, 3] Z-up 坐标系
                    normals = part_data['normals']  # [P, 3] Z-up 坐标系
                else:
                    raise ValueError(f"无法找到点云数据，可用键: {list(part_data.keys())}")
                
                # 确保形状正确
                if points.shape[1] != 3 or normals.shape[1] != 3:
                    raise ValueError(f"点云数据形状不正确: points={points.shape}, normals={normals.shape}")
                
                # 坐标转换：从 Z-up 转换为 Y-up
                # 输入：x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 2], Z-up, 中心点 (0, 0, 1)
                # 输出：x, y, z ∈ [0, 1], Y-up, 中心点 (0.5, 0.5, 0.5)
                points_y_up = transform_points_z_up_to_y_up(points)  # [P, 3] Y-up 坐标系，[0, 1]
                normals_y_up = transform_normals_z_up_to_y_up(normals)  # [P, 3] Y-up 坐标系
                
                # 拼接点和法线
                part_surface = np.concatenate([points_y_up, normals_y_up], axis=1)  # [P, 6]
            else:
                # 直接是数组（假设已经是转换后的格式）
                part_surface = np.array(part_data)
                if part_surface.shape[1] != 6:
                    raise ValueError(f"Part 数据形状不正确: {part_surface.shape}")
                # 如果输入数组是 Z-up 格式，需要进行转换
                # 这里假设如果直接是数组，可能是已经转换过的，但为了安全，可以添加选项
                # 暂时假设已经是正确格式，如果需要转换，可以拆分为 points 和 normals 后再转换
            
            # 转换为 torch tensor
            part_surface = torch.FloatTensor(part_surface)
            return part_surface
        except Exception as e:
            print(f"加载 part 表面数据失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认数据
            default_surface = torch.zeros(self.pc_size, 6)
            return default_surface
    
    def _load_surface_sdf_points(self, rng, part_surface):
        """处理表面数据，添加标签维度"""
        # part_surface: [P, 6] -> [pc_size, 7]
        if part_surface.shape[0] < self.pc_size:
            # 如果点数不够，重复采样
            replace = True
        else:
            replace = False
        
        ind = rng.choice(part_surface.shape[0], self.pc_size, replace=replace)
        sampled_surface = part_surface[ind]  # [pc_size, 6]
        
        # 添加标签维度（全部为0，因为 pc_sharpedge_size=0）
        if self.sharpedge_label:
            sharpedge_label = np.zeros((self.pc_size, 1))
            sampled_surface = np.concatenate([sampled_surface, sharpedge_label], axis=1)
        
        return torch.FloatTensor(sampled_surface)
    
    def __len__(self):
        return len(self.batched_configs)
    
    def __getitem__(self, idx):
        batch_configs = self.batched_configs[idx]
        
        all_images = []
        all_surfaces = []
        num_parts_list = []
        
        for config in batch_configs:
            try:
                # 加载点云数据
                surface_path = config['surface_path']
                data = np.load(surface_path, allow_pickle=True).item()
                
                # 获取 parts 数据 - 参考 analyze_partcrafter_data.py 的逻辑
                if 'parts' in data and len(data['parts']) > 0:
                    parts = data['parts']
                else:
                    # 如果没有 parts，使用整个物体
                    parts = [data['object']]
                
                # 加载图像
                img_dir = config['hunyuan_images_path']
                img_idx = np.random.randint(0, 24)  # 随机选择一张图像
                img_path = os.path.join(img_dir, f'{img_idx:03d}.png')
                image = self._load_image(img_path)
                
                # 处理每个 part
                for part in parts:
                    part_surface = self._load_part_surface(part)  # [P, 6]
                    # 处理表面数据
                    rng = np.random.default_rng()
                    processed_surface = self._load_surface_sdf_points(rng, part_surface)  # [pc_size, 7]
                    
                    all_surfaces.append(processed_surface)
                    all_images.append(image)
                
                num_parts_list.append(len(parts))
                
            except Exception as e:
                print(f"加载配置失败 {config.get('file', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                # 跳过失败的配置
                continue
        
        # 确保有数据
        if not all_surfaces:
            raise ValueError(f"批次 {idx} 没有成功加载任何数据")
        
        return {
            "images": torch.stack(all_images),        # [N, H, W, 3]
            "part_surfaces": torch.stack(all_surfaces), # [N, pc_size, 7]
            "num_parts": torch.tensor(num_parts_list)  # [M]
        }


class PartCrafterAlignedShapeLatentModule(LightningDataModule):
    """PartCrafter 风格的 PyTorch Lightning 数据模块"""
    
    def __init__(
        self,
        batch_size: int = 4,  # 总 part 数
        num_workers: int = 4,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        cond_stage_key: str = "image",
        image_size: int = 518,
        mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
        std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        sharpedge_label: bool = True,
        return_normal: bool = True, 
        padding: bool = True,
        padding_ratio_range: List[float] = [1.15, 1.15]
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        
        self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # 图像变换
        self.train_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.val_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
    def train_dataloader(self):
        """训练数据加载器"""
        dataset = BatchedPartCrafterDataset(
            data_list=self.train_data_list,
            batch_size=self.batch_size,
            image_transform=self.train_image_transform,
            pc_size=self.pc_size,
            pc_sharpedge_size=self.pc_sharpedge_size,
            sharpedge_label=self.sharpedge_label,
            return_normal=self.return_normal,
            padding=self.padding,
            padding_ratio_range=self.padding_ratio_range
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # DataLoader 的 batch_size=1，因为已经预打包
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: x[0]  # 直接返回
        )

    def val_dataloader(self):
        """验证数据加载器"""
        dataset = BatchedPartCrafterDataset(
            data_list=self.val_data_list,
            batch_size=self.batch_size,
            image_transform=self.val_image_transform,
            pc_size=self.pc_size,
            pc_sharpedge_size=self.pc_sharpedge_size,
            sharpedge_label=self.sharpedge_label,
            return_normal=self.return_normal,
            padding=self.padding,
            padding_ratio_range=self.padding_ratio_range
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # DataLoader 的 batch_size=1，因为已经预打包
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: x[0]  # 直接返回
        )


class AlignedShapeLatentModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        cond_stage_key: str = "all",
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False, 
        padding = True,
        padding_ratio_range=[1.15, 1.15]
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        
        self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
    def train_dataloader(self):
        asl_params = {
            "data_list": self.train_data_list,
            "cond_stage_key": self.cond_stage_key,
            "image_transform": self.train_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range
        }
        dataset = AlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        asl_params = {
            "data_list": self.val_data_list,
            "cond_stage_key": self.cond_stage_key,
            "image_transform": self.val_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal, 
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range
        }
        dataset = AlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )


class MultiPartAlignedShapeLatentModule(LightningDataModule):
    """多 part 数据的 PyTorch Lightning 数据模块"""
    
    def __init__(
        self,
        batch_size: int = 4,  # 每个批次包含的 parts 数量
        num_workers: int = 4,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        cond_stage_key: str = "image",
        image_size: int = 518,
        mean: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
        std: Union[List[float], Tuple[float]] = (0.5, 0.5, 0.5),
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        sharpedge_label: bool = True,
        return_normal: bool = True, 
        padding: bool = True,
        padding_ratio_range: List[float] = [1.15, 1.15]
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        
        self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # 图像变换
        self.train_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.val_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        
    def train_dataloader(self):
        """训练数据加载器"""
        asl_params = {
            "data_list": self.train_data_list,
            "image_transform": self.train_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "batch_size": self.batch_size,  # 传入 batch_size 进行预批次打包
        }
        dataset = MultiPartAlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # DataLoader 的 batch_size=1，因为已经预打包
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=MultiPartCollateWrapper(self.batch_size)  # 使用可 pickle 的 wrapper
        )

    def val_dataloader(self):
        """验证数据加载器"""
        asl_params = {
            "data_list": self.val_data_list,
            "image_transform": self.val_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "batch_size": self.batch_size,  # 传入 batch_size 进行预批次打包
        }
        dataset = MultiPartAlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # DataLoader 的 batch_size=1，因为已经预打包
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=MultiPartCollateWrapper(self.batch_size)  # 使用可 pickle 的 wrapper
        )
