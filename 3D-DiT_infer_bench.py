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