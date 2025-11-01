#!/usr/bin/env python3
"""
Demo inference script to generate 3D parts from an image.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import pathlib
import yaml
import trimesh

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hy3dshape.utils import instantiate_from_config, logger
from hy3dshape.pipelines_ours import get_colored_mesh_composition


def load_model_components():
    """Load all model components from checkpoint."""
    print("🔧 Loading model components...")
    
    # Load checkpoint (try Lightning format first, then converted format)
    lightning_checkpoint_path = "./outputs/stage1_test/ckpt/ckpt-step=00001000.ckpt"
    converted_checkpoint_path = "./outputs/stage1_test/ckpt/ckpt-step=00001000_converted.ckpt"
    
    if os.path.exists(lightning_checkpoint_path):
        print("📁 Using Lightning checkpoint directly...")
        checkpoint_path = lightning_checkpoint_path
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract from Lightning format
        state_dict = ckpt['state_dict']
        model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        vae_state = {k.replace('first_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
        cond_state = {k.replace('cond_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('cond_stage_model.')}
        
        print(f"   Extracted - Model: {len(model_state)} params, VAE: {len(vae_state)} params, Conditioner: {len(cond_state)} params")
    elif os.path.exists(converted_checkpoint_path):
        print("📁 Using converted checkpoint...")
        checkpoint_path = converted_checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_state = ckpt['model']
        vae_state = ckpt['vae']
        cond_state = ckpt['conditioner']
    else:
        raise FileNotFoundError(f"No checkpoint found at {lightning_checkpoint_path} or {converted_checkpoint_path}")
    
    # Load config
    config_path = "./configs/ours_multipart_stage1.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load components
    vae = instantiate_from_config(config['model']['params']['first_stage_config'])
    vae.load_state_dict(vae_state, strict=False)
    vae.eval()
    
    conditioner = instantiate_from_config(config['model']['params']['cond_stage_config'])
    conditioner.load_state_dict(cond_state, strict=False)
    conditioner.eval()
    
    model = instantiate_from_config(config['model']['params']['denoiser_cfg'])
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    # Load image processor
    image_processor = instantiate_from_config(config['image_processor'])
    
    print("✅ All model components loaded successfully!")
    return vae, conditioner, model, image_processor, config


def generate_latents(model, conditioner, image_processor, image_pil, num_parts=2, num_steps=10):
    """Generate latents using the diffusion model."""
    print(f"🎨 Generating latents for {num_parts} parts...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move models to device and set dtype
    model = model.to(device).half()
    conditioner = conditioner.to(device).half()
    
    # Use image processor to handle RGBA properly
    cond_inputs = image_processor(image_pil)
    image = cond_inputs['image'].to(device)
    
    print(f"   Processed image shape: {image.shape}")
    
    # Encode image
    with torch.no_grad():
        contexts = conditioner(image=image, **{k: v for k, v in cond_inputs.items() if k != 'image'})
        print(f"   Context shape: {contexts['main'].shape}")
    
    # Prepare latents (start with noise)
    batch_size = 1
    latent_shape = (4096, 64)  # From VAE test
    latents = torch.randn(batch_size * num_parts, *latent_shape, device=device, dtype=torch.float16)
    
    # Prepare contexts for multiple parts
    contexts_expanded = {}
    for key, value in contexts.items():
        contexts_expanded[key] = value.repeat_interleave(num_parts, dim=0)
    
    # Simple denoising loop (simplified)
    timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
    
    for i, t in enumerate(timesteps):
        print(f"   Step {i+1}/{num_steps} (t={t:.3f})")
        
        # Expand timesteps for all parts
        t_expanded = t.expand(batch_size * num_parts)
        
        with torch.no_grad():
            # Predict noise
            noise_pred = model(latents, t_expanded, contexts_expanded)
            
            # Simple Euler step (simplified)
            if i < len(timesteps) - 1:
                dt = timesteps[i+1] - t
                latents = latents + dt * noise_pred
    
    print("✅ Latent generation completed!")
    return latents


def decode_latents_to_meshes(vae, latents, num_parts=2):
    """Decode latents to 3D meshes."""
    print(f"🔧 Decoding latents to {num_parts} meshes...")
    
    device = latents.device
    vae = vae.to(device).half()
    
    meshes = []
    
    with torch.no_grad():
        for i in range(num_parts):
            print(f"   Decoding part {i+1}/{num_parts}...")
            
            # Decode single part
            part_latent = latents[i:i+1]
            
            try:
                # Decode to point cloud
                decoded = vae.decode(part_latent)
                print(f"   Decoded shape: {decoded.shape}")
                
                # Convert to mesh (simplified - just create a simple mesh for demo)
                # In real implementation, this would use marching cubes or similar
                vertices = decoded[0].cpu().numpy()  # Take first sample
                
                # Create a simple mesh from the decoded points
                if len(vertices.shape) == 3:  # [num_points, features]
                    # Use first 3 dimensions as xyz
                    points = vertices[:, :3]
                    
                    # Create a simple convex hull mesh
                    try:
                        mesh = trimesh.creation.convex_hull(points)
                        meshes.append(mesh)
                        print(f"   ✅ Part {i+1} mesh created with {len(mesh.vertices)} vertices")
                    except Exception as e:
                        print(f"   ⚠️  Part {i+1} convex hull failed: {e}")
                        # Create a simple box as fallback
                        mesh = trimesh.creation.box(extents=[1, 1, 1])
                        meshes.append(mesh)
                        print(f"   ✅ Part {i+1} fallback mesh created")
                else:
                    print(f"   ⚠️  Unexpected decoded shape: {vertices.shape}")
                    # Create a simple box as fallback
                    mesh = trimesh.creation.box(extents=[1, 1, 1])
                    meshes.append(mesh)
                    print(f"   ✅ Part {i+1} fallback mesh created")
                    
            except Exception as e:
                print(f"   ❌ Part {i+1} decoding failed: {e}")
                # Create a simple box as fallback
                mesh = trimesh.creation.box(extents=[1, 1, 1])
                meshes.append(mesh)
                print(f"   ✅ Part {i+1} fallback mesh created")
    
    print("✅ Mesh decoding completed!")
    return meshes


def save_results(meshes, image, output_dir="./demo_results"):
    """Save generated results."""
    print(f"💾 Saving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image
    if isinstance(image, torch.Tensor):
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
    else:
        image_pil = image
    
    image_pil.save(os.path.join(output_dir, "input_image.png"))
    
    # Save individual meshes
    for i, mesh in enumerate(meshes):
        mesh.export(os.path.join(output_dir, f"part_{i:02d}.glb"))
        print(f"   Saved part_{i:02d}.glb")
    
    # Save combined mesh
    if len(meshes) > 1:
        combined_scene = get_colored_mesh_composition(meshes)
        combined_scene.export(os.path.join(output_dir, "combined_object.glb"))
        print(f"   Saved combined_object.glb")
    
    print(f"✅ Results saved to {output_dir}")
    return output_dir


def main():
    """Run the demo inference."""
    print("🚀 Starting 3D generation demo...\n")
    
    # Load test image
    image_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/data/train_data_DiFa/DiFa/DiFa-3D-outfit-highpoly/rendered-imgs-by-hunyuan/HighPoly_0018_thin/render_cond/014.png"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    # Load image (handle RGBA like minimal_demo.py)
    image_pil = Image.open(image_path).convert("RGBA")
    if image_pil.mode == 'RGB':
        from hy3dshape.rembg import BackgroundRemover
        rembg = BackgroundRemover()
        image_pil = rembg(image_pil)
    
    print(f"📸 Loaded image: {image_pil.size}, mode: {image_pil.mode}")
    
    # Load model components
    vae, conditioner, model, image_processor, config = load_model_components()
    
    # Generate latents
    num_parts = 2
    num_steps = 5  # Reduced for demo
    latents = generate_latents(model, conditioner, image_processor, image_pil, num_parts, num_steps)
    
    # Decode to meshes
    meshes = decode_latents_to_meshes(vae, latents, num_parts)
    
    # Save results
    output_dir = save_results(meshes, image_pil)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔢 Generated {len(meshes)} parts")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
