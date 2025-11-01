#!/usr/bin/env python3
"""
Simple multi-part 3D generation demo using our trained checkpoint.
Similar to minimal_demo.py but with multi-part support.
"""

from trimesh.base import Trimesh


import os
import sys
import torch
from PIL import Image

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines_ours import Hunyuan3DMultiPartPipeline

def main(use_pretrained=True):
    """Run the multi-part 3D generation demo."""
    print("🚀 Starting multi-part 3D generation demo...\n")
    
    if use_pretrained:
        # Use original pretrained weights
        checkpoint_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/models/pretrained/Hunyuan3D-2.1"
        config_path = None  # Not needed for pretrained weights
        print("🔧 Loading original pretrained pipeline...")
    else:
        # Use our trained weights
        checkpoint_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/outputs/stage1_test/ckpt/ckpt-step=00001000.ckpt"
        config_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/configs/ours_multipart_stage1.yaml"
        print("🔧 Loading our trained pipeline...")
    
    pipeline_shapegen = Hunyuan3DMultiPartPipeline.from_our_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda',
        dtype=torch.float16
    )
    print("✅ Pipeline loaded successfully!")
    
    # Load test image (exactly like minimal_demo.py)
    image_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/demos/demo.png"
    num_parts = 2 # item["num_parts"]
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    print(f"📸 Loaded image: {image.size}, mode: {image.mode}")
    
    # Generate 3D with multiple parts (new num_parts parameter)
    print("🎨 Generating 3D mesh with 2 parts...")
    meshes = pipeline_shapegen(
        image=[image] * num_parts,  # Test with single part first       
        num_inference_steps=50, # Use sufficient steps for quality
        guidance_scale=5.0     # Disable classifier-free guidance for testing
    )
    print("✅ 3D generation completed!")
    
    # Save individual parts
    if isinstance(meshes, list):
        print(f"📦 Generated {len(meshes)} meshes")
        for i, mesh in enumerate(meshes):
            if hasattr(mesh, 'export'):
                output_path = f"part_{i}.glb"
                mesh.export(output_path)
                print(f"💾 Saved {output_path}")
            else:
                print(f"⚠️  Part {i} is not a valid mesh object: {type(mesh)}")
    else:
        print(f"⚠️  Unexpected return type: {type(meshes)}")
    
    # # Optional: Generate and save combined object with colors
    # print("🎨 Generating combined object...")
    # combined_scene, part_meshes = pipeline_shapegen.generate_complete_object(
    #     image=image,
    #     num_parts=2,
    #     num_inference_steps=50,
    #     guidance_scale=5.0
    # )
    # combined_scene.export("combined_object.glb")
    # print("💾 Saved combined_object.glb")
    
    # print("✅ Multi-part 3D generation completed!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-part 3D generation demo')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='Use original pretrained weights (default: True)')
    parser.add_argument('--use-trained', action='store_true', default=False,
                        help='Use our trained weights')
    
    args = parser.parse_args()
    
    # Determine which weights to use
    use_pretrained = args.use_pretrained and not args.use_trained
    
    print(f"🎯 Using {'pretrained' if use_pretrained else 'trained'} weights")
    success = main(use_pretrained=use_pretrained)
    sys.exit(0 if success else 1)