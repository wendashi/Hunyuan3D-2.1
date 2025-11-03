#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速诊断数据加载问题
"""

import os
import sys
import json
import numpy as np
import signal

sys.path.append('/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1')

def timeout_handler(signum, frame):
    raise TimeoutError("数据加载超时！")

# 设置超时
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60秒超时

try:
    print("1. 检查 JSON 文件...")
    json_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train-Test-Set/PartCrafter/train/merged-train_updated-stage1-hunyuan-minimal.json"
    if not os.path.exists(json_path):
        json_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/Train-Test-Set/PartCrafter/train/merged-train_updated-stage1-hunyuan.json"
    
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    print(f"   ✅ JSON 文件读取成功，{len(data_list)} 个样本")
    
    # 检查第一个样本
    sample = data_list[0]
    print(f"\n2. 检查第一个样本:")
    print(f"   surface_path: {sample.get('surface_path')}")
    print(f"   hunyuan_images_path: {sample.get('hunyuan_images_path')}")
    
    # 检查 .npy 文件
    npy_path = sample.get('surface_path')
    if npy_path and os.path.exists(npy_path):
        print(f"   ✅ .npy 文件存在")
        data = np.load(npy_path, allow_pickle=True).item()
        print(f"   数据键: {list(data.keys())}")
        if 'parts' in data:
            print(f"   Parts 数量: {len(data['parts'])}")
    else:
        print(f"   ❌ .npy 文件不存在: {npy_path}")
    
    # 检查图像路径
    img_path = sample.get('hunyuan_images_path')
    if img_path:
        if os.path.exists(img_path):
            print(f"   ✅ 图像目录存在")
            img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
            print(f"   图像文件数量: {len(img_files)}")
            if len(img_files) > 0:
                first_img = os.path.join(img_path, img_files[0])
                print(f"   第一个图像: {first_img}")
                import cv2
                img = cv2.imread(first_img, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    print(f"   图像形状: {img.shape}")
                else:
                    print(f"   ⚠️  无法读取图像")
        else:
            print(f"   ⚠️  图像目录不存在: {img_path}")
    
    print("\n3. 测试直接加载数据集...")
    from hy3dshape.hy3dshape.data.dit_asl_ours import MultiPartAlignedShapeLatentDataset
    from torchvision import transforms
    
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = MultiPartAlignedShapeLatentDataset(
        data_list=json_path,
        image_transform=image_transform,
        pc_size=81920,
        pc_sharpedge_size=0,
        sharpedge_label=True,
        return_normal=True,
        image_size=518
    )
    print(f"   ✅ 数据集创建成功，{len(dataset)} 个对象")
    
    print("\n4. 测试加载第一个样本...")
    import time
    start = time.time()
    parts_data = dataset[0]
    elapsed = time.time() - start
    print(f"   ✅ 第一个样本加载成功，耗时: {elapsed:.2f} 秒")
    print(f"   Parts 数量: {len(parts_data)}")
    
    for i, part in enumerate(parts_data):
        print(f"     Part {i}: surface shape={part['surface'].shape}, image shape={part['image'].shape}")
    
    signal.alarm(0)  # 取消超时
    print("\n✅ 所有检查通过！")
    
except TimeoutError as e:
    print(f"\n❌ {e}")
    print("数据加载卡住了！可能的原因：")
    print("  1. 图像文件读取很慢或卡住")
    print("  2. 点云处理时间过长")
    print("  3. 某个文件损坏或格式错误")
    sys.exit(1)
except Exception as e:
    signal.alarm(0)
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

