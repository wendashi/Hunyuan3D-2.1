import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
# 设置环境变量，强制使用 GPU 6
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
from PIL import Image
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
import tqdm

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

output_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0620'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# # paint
max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth.1"
conf.multiview_cfg_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.multiview_pretrained_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1"
conf.custom_pipeline         = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = f'{output_path}/demo_textured.glb'
output_mesh_path = paint_pipeline(
    mesh_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0619/demo.glb", 
    image_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/assets/demo.png',
    output_mesh_path = output_mesh_path
)

# mesh_paths = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0809-Tpose'
# image_paths = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-0809-Tpose'