# Hunyuan 3D Multi-Part Pipeline for PartCrafter-style inference
# Based on Hunyuan3D-2.1 and PartCrafter inference mechanisms

import copy
import importlib
import inspect
import os
from typing import List, Optional, Union
import PIL
import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model
from .pipelines import Hunyuan3DDiTFlowMatchingPipeline, retrieve_timesteps, export_to_trimesh


def get_colored_mesh_composition(
    meshes: List[trimesh.Trimesh],
    is_random: bool = True,
    is_sorted: bool = False, 
    RGB: List[tuple] = None
):
    """
    Combine multiple part meshes into a single colored scene.
    Adapted from PartCrafter's get_colored_mesh_composition function.
    """
    if RGB is None:
        # Default color palette
        RGB = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 192, 192), (128, 128, 128), (255, 165, 0), (255, 20, 147)
        ]
    
    if is_sorted:
        volumes = []
        for mesh in meshes:
            try:
                volume = mesh.volume
            except:
                volume = 0.0
            volumes.append(volume)
        # sort by volume from large to small
        meshes = [x for _, x in sorted(zip(volumes, meshes), key=lambda pair: pair[0], reverse=True)]
    
    colored_scene = trimesh.Scene()
    for idx, mesh in enumerate(meshes):
        if is_random:
            color = (np.random.rand(3) * 256).astype(int)
        else:
            color = np.array(RGB[idx % len(RGB)])
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=color,
        )
        colored_scene.add_geometry(mesh)
    return colored_scene


class Hunyuan3DMultiPartPipeline(Hunyuan3DDiTFlowMatchingPipeline):
    """
    Multi-part 3D generation pipeline based on Hunyuan3D-2.1.
    Supports generating multiple parts from a single image and combining them into a complete object.
    """

    @classmethod
    def from_our_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = None,
        device='cuda',
        dtype=torch.float16,
    ):
        """
        Load pipeline from checkpoint with automatic format detection.
        Supports both PyTorch Lightning checkpoints and original pretrained weights.
        
        Args:
            checkpoint_path: Path to the checkpoint (Lightning or pretrained)
            config_path: Path to the training config YAML file (only needed for Lightning checkpoints)
            device: Device to load models on
            dtype: Data type for model weights
            
        Returns:
            Hunyuan3DMultiPartPipeline instance
        """
        import pathlib
        from hy3dshape.utils import instantiate_from_config
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        
        # Detect checkpoint format
        try:
            torch.serialization.add_safe_globals([pathlib.PosixPath])
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Check if it's a Lightning checkpoint
            if 'state_dict' in ckpt:
                print("🔧 Detected PyTorch Lightning checkpoint format")
                return cls._from_lightning_checkpoint(ckpt, config_path, device, dtype)
            else:
                print("🔧 Detected original pretrained weights format")
                return cls._from_pretrained_weights(checkpoint_path, device, dtype)
                
        except Exception as e:
            print(f"⚠️  Failed to load as checkpoint, trying as pretrained model path: {e}")
            return cls._from_pretrained_weights(checkpoint_path, device, dtype)
    
    @classmethod
    def _from_lightning_checkpoint(
        cls,
        ckpt,
        config_path: str,
        device='cuda',
        dtype=torch.float16,
    ):
        """Load from PyTorch Lightning checkpoint format."""
        import yaml
        from hy3dshape.utils import instantiate_from_config
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract from Lightning format
        state_dict = ckpt['state_dict']
        model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        vae_state = {k.replace('first_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
        cond_state = {k.replace('cond_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('cond_stage_model.')}
        
        print(f"   Extracted - Model: {len(model_state)} params, VAE: {len(vae_state)} params, Conditioner: {len(cond_state)} params")
        
        # Load components using our config structure
        vae = instantiate_from_config(config['model']['params']['first_stage_config'])
        vae.load_state_dict(vae_state, strict=False)
        vae.eval()
        
        conditioner = instantiate_from_config(config['model']['params']['cond_stage_config'])
        conditioner.load_state_dict(cond_state, strict=False)
        conditioner.eval()
        
        model = instantiate_from_config(config['model']['params']['denoiser_cfg'])
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        image_processor = instantiate_from_config(config['model']['params']['image_processor_cfg'])
        
        # Create scheduler using the same config as pretrained model
        scheduler = instantiate_from_config({
            'target': 'hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler',
            'params': {
                'num_train_timesteps': 1000
            }
        })
        
        # Move to device
        vae = vae.to(device).to(dtype)
        conditioner = conditioner.to(device).to(dtype)
        model = model.to(device).to(dtype)
        
        # Create pipeline instance
        return cls(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
    
    @classmethod
    def _from_pretrained_weights(
        cls,
        model_path: str,
        device='cuda',
        dtype=torch.float16,
    ):
        """Load from original pretrained weights using Hunyuan3DDiTFlowMatchingPipeline."""
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        
        print(f"🔧 Loading pretrained model from: {model_path}")
        
        # Check if it's a directory (pretrained model) or a file (checkpoint)
        import os
        if os.path.isdir(model_path):
            # It's a pretrained model directory, use from_pretrained
            original_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        else:
            # It's a checkpoint file, use from_single_file with config_path
            # Use the converted config for inference
            config_path = "/opt/liblibai-models/user-workspace/colabrate/wenda/projects-3d/Hunyuan3D-2.1/hy3dshape/configs/ours_converted_for_inference.yaml"
            original_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                ckpt_path=model_path,
                config_path=config_path
            )
        
        # Extract components from the original pipeline
        vae = original_pipeline.vae
        model = original_pipeline.model
        scheduler = original_pipeline.scheduler
        conditioner = original_pipeline.conditioner
        image_processor = original_pipeline.image_processor
        
        # Move to device
        vae = vae.to(device).to(dtype)
        conditioner = conditioner.to(device).to(dtype)
        model = model.to(device).to(dtype)
        
        # Create our multi-part pipeline instance
        return cls(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_parts: int = 1,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        **kwargs,
    ) -> List[trimesh.Trimesh]:
        """
        Generate multiple 3D parts from a single image.
        
        Args:
            image: Input image
            num_parts: Number of parts to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            generator: Random generator for reproducibility
            box_v: Bounding box size
            octree_resolution: Octree resolution for mesh extraction
            mc_level: Marching cubes level
            mc_algo: Marching cubes algorithm
            num_chunks: Number of chunks for processing
            output_type: Output type ("trimesh" or "latent")
            enable_pbar: Enable progress bar
            mask: Optional mask
            
        Returns:
            List of generated part meshes
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # Prepare image input
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        
        # Encode condition
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export_multipart(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )


    def _export_multipart(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        """
        Export latents to multiple part meshes.
        """
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            
            # Process each latent separately (like PartCrafter)
            all_meshes = []
            with tqdm(total=latents.shape[0], disable=not enable_pbar, desc="Decoding parts") as pbar:
                for i in range(latents.shape[0]):
                    try:
                        mesh_output = self.vae.latents2mesh(
                            latents[i:i+1],  # Single latent
                            bounds=box_v,
                            mc_level=mc_level,
                            num_chunks=num_chunks,
                            octree_resolution=octree_resolution,
                            mc_algo=mc_algo,
                            enable_pbar=False,
                        )
                        if output_type == 'trimesh':
                            mesh_output = export_to_trimesh(mesh_output)
                            # 确保 mesh_output 是单个 trimesh.Trimesh 对象
                            if isinstance(mesh_output, list):
                                if len(mesh_output) > 0:
                                    mesh_output = mesh_output[0]  # 取第一个元素
                                else:
                                    # 如果列表为空，创建 dummy mesh
                                    mesh_output = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        all_meshes.append(mesh_output)
                    except Exception as e:
                        logger.warning(f"Failed to decode part {i}: {e}")
                        # Create a dummy mesh if decoding fails
                        dummy_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        all_meshes.append(dummy_mesh)
                    pbar.update(1)
        else:
            all_meshes = latents

        return all_meshes

    def generate_complete_object(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_parts: int = 1,
        **kwargs
    ) -> trimesh.Scene:
        """
        Generate a complete 3D object with multiple parts and combine them.
        
        Args:
            image: Input image
            num_parts: Number of parts to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Combined trimesh.Scene with all parts
        """
        # Generate individual parts
        part_meshes = self.__call__(
            image=image,
            num_parts=num_parts,
            **kwargs
        )
        
        # Combine parts into a single scene
        combined_scene = get_colored_mesh_composition(part_meshes)
        
        return combined_scene, part_meshes
