import torch
import json
from pathlib import Path
from pn_models import PointNetMLPJoint
from benchmarks import VanillaDeepONet, SpectralDeepONet, DenseNoFFT

def load_model_with_checkpoint(model_path, model_type, device='cpu'):
    """
    Load a trained model from checkpoint.
    For models without saved architecture, reconstruct from preset.
    """
    ckpt = torch.load(model_path, map_location=device)
    
    # Get architecture from checkpoint or preset
    arch = ckpt.get('arch')
    
    if arch is not None and 'encoder_cfg' in arch:
        # PointNetMLPJoint has architecture saved
        if model_type == 'PointNetMLPJoint':
            model = PointNetMLPJoint(
                encoder_cfg=arch['encoder_cfg'],
                mlp_hidden=arch['head_hidden']
            )
        elif model_type == 'DenseNoFFT':
            model = DenseNoFFT(
                latent_dim=arch['encoder_cfg']['latent_dim'],
                mlp_hidden=arch['head_hidden'],
                encoder_cfg=arch['encoder_cfg']
            )
        else:
            raise ValueError(f"Unexpected model type with arch: {model_type}")
    else:
        # DeepONet models don't save architecture, load from preset
        # Determine which preset was used (they all use "L")
        repo_dir = Path(model_path).parent.parent.parent
        preset_file = Path(model_path).parent.parent / 'model_presets.json'
        
        with open(preset_file, 'r') as f:
            presets = json.load(f)
        
        preset = presets['L']  # All models use L preset
        
        # Build encoder_cfg from preset
        # Default to 4 freqs if not specified (matches training script defaults)
        posenc = preset.get('posenc', {'n_freqs': 4, 'scale': 1.0})
        head_posenc = preset.get('head_posenc', {'n_freqs': 4, 'scale': 1.0})
        
        encoder_cfg = {
            'latent_dim': preset['latent_dim'],
            'pre_hidden': preset['pre_hidden'],
            'sa_blocks': preset['sa_blocks'],
            'gf_hidden': preset['gf_hidden'],
            'norm': preset.get('norm', 'batch'),
            'num_groups': preset.get('num_groups', 16),
            'pool': preset.get('pool', 'max'),
            'posenc': posenc,  # Fourier features for encoder input
        }
        
        # Extract DeepONet parameters from head_hidden
        head_hidden = preset['head_hidden']
        # head_hidden format: [branch_hidden..., basis_dim]
        # e.g., [512, 512, 256] means branch_hidden=[512, 512], basis_dim=256
        if len(head_hidden) > 1:
            do_branch_hidden = head_hidden[:-1]
            do_basis_dim = head_hidden[-1]
        else:
            do_branch_hidden = [256, 256]
            do_basis_dim = 128
        do_trunk_hidden = do_branch_hidden  # symmetric
        
        if model_type == 'SpectralDeepONet':
            # SpectralDeepONet has Fourier features for trunk
            n_freqs = head_posenc.get('n_freqs', 10)
            scale = head_posenc.get('scale', 2.0)
            model = SpectralDeepONet(
                latent_dim=preset['latent_dim'],
                basis_dim=do_basis_dim,
                branch_hidden=do_branch_hidden,
                trunk_hidden=do_trunk_hidden,
                n_freqs=n_freqs,
                scale=scale,
                encoder_cfg=encoder_cfg
            )
        elif model_type == 'VanillaDeepONet':
            model = VanillaDeepONet(
                latent_dim=preset['latent_dim'],
                basis_dim=do_basis_dim,
                branch_hidden=do_branch_hidden,
                trunk_hidden=do_trunk_hidden,
                encoder_cfg=encoder_cfg
            )
        elif model_type == 'DenseNoFFT':
            model = DenseNoFFT(
                latent_dim=preset['latent_dim'],
                mlp_hidden=preset['head_hidden'],
                encoder_cfg=encoder_cfg
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    
    # Extract normalization params
    norm_params = {
        'coord_center': ckpt['coord_center'].to(device),
        'coord_half_range': ckpt['coord_half_range'].to(device),
        'stress_mean': ckpt['stress_mean'].to(device),
        'stress_std': ckpt['stress_std'].to(device)
    }
    
    return model, norm_params, ckpt
