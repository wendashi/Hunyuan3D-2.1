import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
# 设置环境变量，强制使用 GPU 6
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

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

input_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-0619'
output_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0620'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# shape
model_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

for image_path in tqdm.tqdm(os.listdir(input_path)):
    full_image_path = os.path.join(input_path, image_path)
    image = Image.open(full_image_path)
    
    # 先检查原始图片模式，如果是RGB才需要去背景
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # 确保最终是RGBA模式
    if image.mode != 'RGBA':
        image = image.convert("RGBA")

    mesh = pipeline_shapegen(image=image)[0]
    mesh.export(os.path.join(output_path, image_path.replace('.png', '.glb')))

# # paint
# max_num_view = 6  # can be 6 to 9
# resolution = 512  # can be 768 or 512
# conf = Hunyuan3DPaintConfig(max_num_view, resolution)
# conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
# conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
# conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
# paint_pipeline = Hunyuan3DPaintPipeline(conf)

# output_mesh_path = 'demo_textured.glb'
# output_mesh_path = paint_pipeline(
#     mesh_path = "demo.glb", 
#     image_path = 'assets/demo.png',
#     output_mesh_path = output_mesh_path
# )
