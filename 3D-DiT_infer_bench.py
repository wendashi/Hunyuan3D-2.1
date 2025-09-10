import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
import json
# 设置环境变量，强制使用 GPU 6
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import tqdm
from typing import Tuple, List

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

input_json_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-accessory-0826/test_0826_accessory.json'
# input_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-0809-Tpose'
output_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test_0826_accessory'

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(input_json_path, 'r') as f:
    data = json.load(f)

# shape
model_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)


def human(n): 
    return f"{n/1e9:.1f} B" if n>=1e9 else f"{n/1e6:.1f} M" if n>=1e6 else f"{n/1e3:.1f} K" if n>=1e3 else str(n)

def stat(m: torch.nn.Module):
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    allp = sum(p.numel() for p in m.parameters())
    return t, allp - t, allp, sum(p.numel()*p.element_size() for p in m.parameters()) + sum(b.numel()*b.element_size() for b in m.buffers())

mods = []
if hasattr(pipeline_shapegen, "model"):             mods.append(("model",             pipeline_shapegen.model))
if hasattr(pipeline_shapegen, "conditioner"):  mods.append(("cond_stage_model",  pipeline_shapegen.conditioner))
if hasattr(pipeline_shapegen, "vae"): mods.append(("first_stage_model", pipeline_shapegen.vae))

print("| Name              | Type               | Params")
print("---------------------------------------------------------")
tot_tr = tot_nt = tot_all = 0; tot_bytes = 0
for name, m in mods:
    tr, nt, allp, bytes_ = stat(m)
    tot_tr += tr; tot_nt += nt; tot_all += allp; tot_bytes += bytes_
    print(f"{name:<1} | {name:<18}| {type(m).__name__:<19}| {human(allp):>7}")

print("---------------------------------------------------------")
print(f"{human(tot_tr):>7}     Trainable params")
print(f"{human(tot_nt):>7}     Non-trainable params")
print(f"{human(tot_all):>7}     Total params")
print(f"{tot_bytes/1e6:,.3f} Total estimated model params size (MB)")


for item in tqdm.tqdm(data):
    image_path = item['image_path']
    category = item.get('category', 'unknown')

    # 直接使用 image_path，因为它已经是完整路径
    image = Image.open(image_path)
    
    # 先检查原始图片模式，如果是RGB才需要去背景
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # 确保最终是RGBA模式
    if image.mode != 'RGBA':
        image = image.convert("RGBA")

    # 保存处理后的图片用于检查效果
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 创建保存处理后图片的目录
    processed_images_dir = os.path.join(output_path, 'processed_images', category)
    os.makedirs(processed_images_dir, exist_ok=True)
    
    # 保存处理后的图片
    processed_image_path = os.path.join(processed_images_dir, f"{name_without_ext}_processed.png")
    image.save(processed_image_path, "PNG")
    print(f"Saved processed image to: {processed_image_path}")

    mesh = pipeline_shapegen(image=image)[0]
    
    # 获取文件名并替换扩展名为 .glb
    output_filename = os.path.splitext(filename)[0] + '.glb'
    mesh.export(os.path.join(output_path, output_filename))