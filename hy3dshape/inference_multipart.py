#!/usr/bin/env python3
"""
Multi-part 3D generation inference script for Hunyuan3D-2.1 with PartCrafter-style multi-part support.
"""

import argparse
import os
import sys
import time
from typing import Union

import numpy as np
import torch
import trimesh
from PIL import Image
from accelerate.utils import set_seed

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hy3dshape.pipelines_ours import Hunyuan3DMultiPartPipeline, get_colored_mesh_composition
from hy3dshape.utils import logger


def load_pipeline_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Hunyuan3DMultiPartPipeline:
    """
    Load the multi-part pipeline from a checkpoint.
    Supports both Lightning and standard checkpoint formats.
    """
    logger.info(f"Loading pipeline from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    import pathlib
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if it's a Lightning checkpoint
    if 'state_dict' in ckpt:
        logger.info("Detected Lightning checkpoint format, extracting components...")
        # Lightning format: extract from state_dict
        state_dict = ckpt['state_dict']
        
        # Separate components
        model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        vae_state = {k.replace('first_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
        cond_state = {k.replace('cond_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('cond_stage_model.')}
        
        logger.info(f"Extracted - Model: {len(model_state)} params, VAE: {len(vae_state)} params, Conditioner: {len(cond_state)} params")
        
        # Create standard checkpoint format
        standard_ckpt = {
            'model': model_state,
            'vae': vae_state,
            'conditioner': cond_state,
        }
    else:
        logger.info("Detected standard checkpoint format, using directly...")
        # Standard format: use directly
        standard_ckpt = ckpt
    
    # Save temporary checkpoint for pipeline loading
    temp_ckpt_path = checkpoint_path.replace('.ckpt', '_temp_for_pipeline.ckpt')
    torch.save(standard_ckpt, temp_ckpt_path)
    
    try:
        # Load the pipeline using the standard checkpoint
        pipeline = Hunyuan3DMultiPartPipeline.from_single_file(
            ckpt_path=temp_ckpt_path,
            config_path=config_path,
            device=device,
            dtype=dtype
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_ckpt_path):
            os.remove(temp_ckpt_path)
    
    return pipeline


def run_inference(
    pipeline: Hunyuan3DMultiPartPipeline,
    image_input: Union[str, Image.Image],
    num_parts: int,
    seed: int = 0,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    box_v: float = 1.01,
    octree_resolution: int = 384,
    mc_level: float = 0.0,
    num_chunks: int = 8000,
    output_type: str = "trimesh",
    enable_pbar: bool = True,
) -> tuple:
    """
    Run multi-part 3D generation inference.
    
    Returns:
        tuple: (combined_scene, part_meshes, processed_image)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Prepare image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        processed_image = Image.open(image_input)
    else:
        processed_image = image_input
    
    logger.info(f"Generating {num_parts} parts from image...")
    start_time = time.time()
    
    # Generate complete object with multiple parts
    combined_scene, part_meshes = pipeline.generate_complete_object(
        image=processed_image,
        num_parts=num_parts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        box_v=box_v,
        octree_resolution=octree_resolution,
        mc_level=mc_level,
        num_chunks=num_chunks,
        output_type=output_type,
        enable_pbar=enable_pbar,
    )
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    return combined_scene, part_meshes, processed_image


def save_results(
    combined_scene: trimesh.Scene,
    part_meshes: list,
    processed_image: Image.Image,
    output_dir: str,
    tag: str = None
):
    """
    Save the generated results.
    """
    if tag is None:
        tag = time.strftime("%Y%m%d_%H_%M_%S")
    
    export_dir = os.path.join(output_dir, tag)
    os.makedirs(export_dir, exist_ok=True)
    
    # Save input image
    processed_image.save(os.path.join(export_dir, "input_image.png"))
    
    # Save individual parts
    for i, mesh in enumerate(part_meshes):
        if mesh is not None:
            mesh.export(os.path.join(export_dir, f"part_{i:02d}.glb"))
        else:
            logger.warning(f"Part {i} is None, skipping...")
    
    # Save combined object
    combined_scene.export(os.path.join(export_dir, "complete_object.glb"))
    
    # Save as individual meshes for easier viewing
    try:
        # Convert scene to a single mesh
        combined_mesh = combined_scene.dump().sum()
        combined_mesh.export(os.path.join(export_dir, "complete_object_single.glb"))
    except Exception as e:
        logger.warning(f"Failed to create single mesh: {e}")
    
    logger.info(f"Results saved to: {export_dir}")
    return export_dir


def main():
    parser = argparse.ArgumentParser(description="Multi-part 3D generation with Hunyuan3D-2.1")
    
    # Input arguments
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--num_parts", type=int, default=3, help="Number of parts to generate")
    
    # Model arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config")
    
    # Generation arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--box_v", type=float, default=1.01, help="Bounding box size")
    parser.add_argument("--octree_resolution", type=int, default=384, help="Octree resolution")
    parser.add_argument("--mc_level", type=float, default=0.0, help="Marching cubes level")
    parser.add_argument("--num_chunks", type=int, default=8000, help="Number of chunks")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Output tag")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    
    if args.num_parts < 1 or args.num_parts > 16:
        raise ValueError("num_parts must be between 1 and 16")
    
    # Set data type
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Load pipeline
    pipeline = load_pipeline_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=args.device,
        dtype=dtype
    )
    
    # Run inference
    combined_scene, part_meshes, processed_image = run_inference(
        pipeline=pipeline,
        image_input=args.image_path,
        num_parts=args.num_parts,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        box_v=args.box_v,
        octree_resolution=args.octree_resolution,
        mc_level=args.mc_level,
        num_chunks=args.num_chunks,
        enable_pbar=True,
    )
    
    # Save results
    export_dir = save_results(
        combined_scene=combined_scene,
        part_meshes=part_meshes,
        processed_image=processed_image,
        output_dir=args.output_dir,
        tag=args.tag
    )
    
    print(f"✅ Multi-part 3D generation completed!")
    print(f"📁 Results saved to: {export_dir}")
    print(f"🔢 Generated {len(part_meshes)} parts")
    print(f"🎯 Combined object saved as: complete_object.glb")


if __name__ == "__main__":
    main()
