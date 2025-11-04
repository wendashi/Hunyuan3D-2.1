import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
import json
# 设置环境变量，强制使用 GPU 6
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.models.autoencoders.model import ShapeVAE  # 用于设置 num_latents
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

input_json_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-1104/test-1104-outfit.json'
# input_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-0809-Tpose'
output_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-1104-outfit'

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(input_json_path, 'r') as f:
    data = json.load(f)

# load pretrained model
model_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

# load finetuned model(only DiT)
# /opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-highpoly/whole_bs=2_latents=512/ckpt/ckpt-step=00002000-v1.ckpt/ckpt-step=00002000-v1.fp32
# /opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-highpoly/2_latents=4096/ckpt/ckpt-step=00006000.ckpt/converted 
checkpoint_dir = '/opt/liblibai-models/user-workspace/colabrate/wenda/models/trained/DiFa/hunyuan3Ddit-highpoly/2_latents=4096/ckpt/ckpt-step=00006000.ckpt/converted'

# 加载分片权重并只替换 DiT 模型部分
def load_dit_from_sharded_checkpoint(checkpoint_dir, pipeline):
    """
    从分片检查点加载训练过的 DiT 模型权重，只替换 pipeline.model 部分
    
    Args:
        checkpoint_dir: 包含分片权重文件的目录
        pipeline: 已加载的预训练 pipeline
    """
    import json
    from collections import OrderedDict
    
    # 加载分片检查点
    index_path = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
    single_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(index_path):
        # 加载分片检查点
        with open(index_path, "r") as f:
            index = json.load(f)
        
        shard_cache = {}
        model_state_dict = OrderedDict()
        
        # 只提取 model 部分的权重（_forward_module.model.*）
        for param_name, shard_file in index.get("weight_map", {}).items():
            if param_name.startswith("_forward_module.model."):
                shard_path = os.path.join(checkpoint_dir, shard_file)
                if shard_file not in shard_cache:
                    print(f"Loading shard: {shard_file}")
                    shard_cache[shard_file] = torch.load(shard_path, map_location="cpu", weights_only=False)
                
                # 移除 _forward_module.model. 前缀，得到实际的模型键名
                model_key = param_name.replace("_forward_module.model.", "")
                model_state_dict[model_key] = shard_cache[shard_file][param_name]
    elif os.path.exists(single_path):
        # 单个文件的情况
        full_state_dict = torch.load(single_path, map_location="cpu", weights_only=False)
        model_state_dict = OrderedDict()
        for key, value in full_state_dict.items():
            if key.startswith("_forward_module.model."):
                model_key = key.replace("_forward_module.model.", "")
                model_state_dict[model_key] = value
    else:
        raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}")
    
    # 加载权重到 pipeline.model
    if len(model_state_dict) > 0:
        missing_keys, unexpected_keys = pipeline.model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            print(f"⚠ 警告: 加载权重时缺少以下键: {missing_keys[:5]}... (共 {len(missing_keys)} 个)")
        if unexpected_keys:
            print(f"⚠ 警告: 加载权重时发现意外的键: {unexpected_keys[:5]}... (共 {len(unexpected_keys)} 个)")
        print(f"✓ 成功加载 {len(model_state_dict)} 个 DiT 模型权重")
    else:
        print("⚠ 警告: 未找到任何 model 权重")

# 加载训练过的 DiT 权重
load_dit_from_sharded_checkpoint(checkpoint_dir, pipeline_shapegen)

# ============================================
# 设置 VAE 的 num_latents 参数
# ============================================
# 如果需要修改 VAE 的 num_latents，设置 ENABLE_CUSTOM_NUM_LATENTS = True
# 并设置 desired_num_latents 为你想要的值
# 
# 注意：
# 1. num_latents 会影响模型架构（transformer 的 n_ctx 等），改变它可能导致权重不兼容
# 2. 确保你使用的 checkpoint 支持该 num_latents 值
# 3. 常用的值有: 512, 4096
# 4. 如果使用不匹配的 num_latents，可能导致：
#    - Volume 解码数据无效（所有值都在同一侧，marching cubes 无法提取表面）
#    - 出现 "Surface level must be within volume data range" 错误
#    - mesh 为 None，无法导出

ENABLE_CUSTOM_NUM_LATENTS = True  # 设置为 True 以启用自定义 num_latents
desired_num_latents = 4096  # 修改为你想要的值（例如: 512, 4096）

if ENABLE_CUSTOM_NUM_LATENTS:
    # 从预训练模型加载 VAE，并覆盖 num_latents 参数
    try:
        new_vae = ShapeVAE.from_pretrained(
            model_path,
            subfolder='hunyuan3d-vae-v2-1',
            num_latents=desired_num_latents,
            device=pipeline_shapegen.device,
            dtype=pipeline_shapegen.dtype
        )
        # 替换 pipeline 中的 VAE
        pipeline_shapegen.vae = new_vae
        print(f"✓ VAE num_latents 已设置为: {desired_num_latents}")
    except Exception as e:
        print(f"⚠ 警告: 无法使用 num_latents={desired_num_latents} 加载 VAE: {e}")
        print("   使用默认配置的 VAE")


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
        os.environ.setdefault("U2NET_HOME", "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/rembg")
        rembg_session = BackgroundRemover() 

        image = rembg_session(image)
    
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

    try:
        mesh = pipeline_shapegen(image=image)[0]
        
        # 检查 mesh 是否为 None（surface extraction 失败）
        if mesh is None:
            print(f"⚠ 警告: {filename} 的表面提取失败，可能是 num_latents 配置不匹配导致的")
            print(f"   跳过该文件")
            continue
        
        # 获取文件名并替换扩展名为 .glb
        output_filename = os.path.splitext(filename)[0] + '.glb'
        mesh.export(os.path.join(output_path, output_filename))
        print(f"✓ 成功生成: {output_filename}")
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")
        import traceback
        traceback.print_exc()
        continue