#!/usr/bin/env python3
"""
Convert our trained checkpoint to standard Hunyuan3D format for inference.
"""

import os
import sys
import torch
import pathlib
import yaml

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def convert_checkpoint():
    """Convert our Lightning checkpoint to standard format."""
    print("🔄 Converting our checkpoint to standard format...")
    
    # Load our checkpoint
    checkpoint_path = "./outputs/stage1_test/ckpt/ckpt-step=00001000.ckpt"
    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract from Lightning format
    state_dict = ckpt['state_dict']
    model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    vae_state = {k.replace('first_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
    cond_state = {k.replace('cond_stage_model.', ''): v for k, v in state_dict.items() if k.startswith('cond_stage_model.')}
    
    print(f"   Extracted - Model: {len(model_state)} params, VAE: {len(vae_state)} params, Conditioner: {len(cond_state)} params")
    
    # Create standard checkpoint format
    standard_ckpt = {
        'model': model_state,
        'vae': vae_state,
        'conditioner': cond_state,
    }
    
    # Create standard config format
    standard_config = {
        'model': {
            'target': 'hy3dshape.models.denoisers.hunyuandit_ours.HunYuanDiTPlain',
            'params': {
                'input_size': 4096,
                'in_channels': 64,
                'hidden_size': 2048,
                'context_dim': 1024,
                'depth': 21,
                'num_heads': 16,
                'qk_norm': True,
                'text_len': 1370,
                'with_decoupled_ca': False,
                'use_attention_pooling': False,
                'qk_norm_type': 'rms',
                'qkv_bias': False,
                'use_pos_emb': False,
                'num_moe_layers': 6,
                'num_experts': 8,
                'moe_top_k': 2,
            }
        },
        'vae': {
            'target': 'hy3dshape.models.autoencoders.ShapeVAE',
            'params': {
                'num_latents': 4096,
                'embed_dim': 64,
                'num_freqs': 8,
                'include_pi': False,
                'heads': 16,
                'width': 1024,
                'num_encoder_layers': 8,
                'num_decoder_layers': 16,
                'qkv_bias': False,
                'qk_norm': True,
                'scale_factor': 1.0039506158752403,
                'geo_decoder_mlp_expand_ratio': 4,
                'geo_decoder_downsample_ratio': 1,
                'geo_decoder_ln_post': True,
                'point_feats': 4,
                'pc_size': 81920,
                'pc_sharpedge_size': 0,
            }
        },
        'conditioner': {
            'target': 'hy3dshape.models.conditioner.SingleImageEncoder',
            'params': {
                'main_image_encoder': {
                    'type': 'DinoImageEncoder',
                    'kwargs': {
                        'config': {
                            'attention_probs_dropout_prob': 0.0,
                            'drop_path_rate': 0.0,
                            'hidden_act': 'gelu',
                            'hidden_dropout_prob': 0.0,
                            'hidden_size': 1024,
                            'image_size': 518,
                            'initializer_range': 0.02,
                            'layer_norm_eps': 1.e-6,
                            'layerscale_value': 1.0,
                            'mlp_ratio': 4,
                            'model_type': 'dinov2',
                            'num_attention_heads': 16,
                            'num_channels': 3,
                            'num_hidden_layers': 24,
                            'patch_size': 14,
                            'qkv_bias': True,
                            'torch_dtype': 'float32',
                            'use_swiglu_ffn': False,
                        },
                        'image_size': 518,
                        'use_cls_token': True,
                    }
                }
            }
        },
        'scheduler': {
            'target': 'hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler',
            'params': {
                'num_train_timesteps': 1000,
            }
        },
        'image_processor': {
            'target': 'hy3dshape.preprocessors.ImageProcessorV2',
            'params': {
                'size': 512,
                'border_ratio': 0.15,
            }
        },
        'pipeline': {
            'target': 'hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline',
        }
    }
    
    # Save converted checkpoint
    converted_ckpt_path = "./outputs/stage1_test/ckpt/ckpt-step=00001000_converted.ckpt"
    torch.save(standard_ckpt, converted_ckpt_path)
    print(f"💾 Saved converted checkpoint: {converted_ckpt_path}")
    
    # Save converted config
    converted_config_path = "./configs/ours_converted_for_inference.yaml"
    with open(converted_config_path, 'w') as f:
        yaml.dump(standard_config, f, default_flow_style=False)
    print(f"💾 Saved converted config: {converted_config_path}")
    
    print("✅ Conversion completed!")
    return converted_ckpt_path, converted_config_path

if __name__ == "__main__":
    convert_checkpoint()
