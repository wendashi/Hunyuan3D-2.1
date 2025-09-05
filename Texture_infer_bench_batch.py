import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import os
import glob
import time
from pathlib import Path
# 设置环境变量，强制使用 GPU 5
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def batch_texture_generation():
    """批量纹理生成函数"""
    
    # 配置路径
    mesh_dir = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test_0826_accessory'
    image_dir = '/opt/liblibai-models/user-workspace/colabrate/wenda/data/test_data/Fashion3D/test-accessory-0826'
    output_dir = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test_0826_accessory-textured'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型配置
    max_num_view = 6  # can be 6 to 9
    resolution = 512  # can be 768 or 512
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth.1"
    conf.multiview_cfg_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.multiview_pretrained_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    
    print("正在加载模型...")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    print("模型加载完成！")
    
    # 获取所有输入文件
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.glb")))
    # 支持多种图片格式
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)
    
    print(f"找到 {len(mesh_files)} 个模型文件")
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 匹配文件对（基于文件名）
    file_pairs = []
    for mesh_file in mesh_files:
        mesh_basename = Path(mesh_file).stem  # 例如 "img_1"
        
        # 查找对应的图片文件
        matching_image = None
        for image_file in image_files:
            image_basename = Path(image_file).stem
            if mesh_basename == image_basename:
                matching_image = image_file
                break
        
        if matching_image:
            file_pairs.append((mesh_file, matching_image, mesh_basename))
        else:
            print(f"警告: 未找到与 {mesh_basename}.glb 匹配的图片文件")
    
    print(f"匹配到 {len(file_pairs)} 对文件")
    
    # 批量处理
    success_count = 0
    error_count = 0
    
    for i, (mesh_path, image_path, basename) in enumerate(tqdm.tqdm(file_pairs, desc="批量生成纹理")):
        try:
            start_time = time.time()
            
            # 输出路径
            output_mesh_path = os.path.join(output_dir, f"{basename}_textured.glb")
            
            print(f"\n[{i+1}/{len(file_pairs)}] 处理: {basename}")
            print(f"  模型: {mesh_path}")
            print(f"  图片: {image_path}")
            print(f"  输出: {output_mesh_path}")
            
            # 检查输入文件是否存在
            if not os.path.exists(mesh_path):
                print(f"  错误: 模型文件不存在: {mesh_path}")
                error_count += 1
                continue
                
            if not os.path.exists(image_path):
                print(f"  错误: 图片文件不存在: {image_path}")
                error_count += 1
                continue
            
            # 跳过已存在的输出文件（可选）
            if os.path.exists(output_mesh_path):
                print(f"  跳过: 输出文件已存在")
                continue
            
            # 执行纹理生成
            result_path = paint_pipeline(
                mesh_path=mesh_path,
                image_path=image_path,
                output_mesh_path=output_mesh_path
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"  成功! 耗时: {processing_time:.2f}秒")
            print(f"  结果保存到: {result_path}")
            success_count += 1
            
            # 强制垃圾回收，释放显存
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  错误: 处理 {basename} 时发生异常: {str(e)}")
            error_count += 1
            # 出错时也清理显存
            torch.cuda.empty_cache()
            continue
    
    # 总结
    print(f"\n" + "="*50)
    print(f"批量处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败数量: {error_count} 个文件")
    print(f"总计文件: {len(file_pairs)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"="*50)

def single_test():
    """单个文件测试（保留原始功能）"""
    output_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0620-single'
    os.makedirs(output_path, exist_ok=True)
        
    # 模型配置
    max_num_view = 6  # can be 6 to 9
    resolution = 512  # can be 768 or 512
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth.1"
    conf.multiview_cfg_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.multiview_pretrained_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    output_mesh_path = f'{output_path}/demo_textured.glb'
    output_mesh_path = paint_pipeline(
        mesh_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/results/hunyuan3D21/test-0619/demo.glb", 
        image_path = '/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/assets/demo.png',
        output_mesh_path = output_mesh_path
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hunyuan3D 纹理生成工具")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch", 
                       help="运行模式: single(单个测试) 或 batch(批量处理)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        print("运行单个文件测试...")
        single_test()
    else:
        print("运行批量处理...")
        batch_texture_generation() 