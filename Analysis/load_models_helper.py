import re
import torch
import json
from pathlib import Path
from pn_models import PointNetMLPJoint
from benchmarks import (
    VanillaDeepONet, SpectralDeepONet, DenseNoFFT,
    ScaledDiagramDeepONet, PointDeepONet,
)


def _find_preset_for_checkpoint(model_path, presets):
    """Find the preset that matches the checkpoint filename by model_name field."""
    stem = Path(model_path).stem  # e.g. 'L-pn2_nosdf_m_febe42d8'
    # Strip geometry prefix (L- or H-)
    basename = re.sub(r'^[LH]-', '', stem)
    # Strip 8-char hex hash suffix (md5 hexdigest[:8], always lowercase, but be flexible)
    model_name_from_file = re.sub(r'_[0-9a-fA-F]{6,}$', '', basename)
    for preset in presets.values():
        if preset.get('model_name') == model_name_from_file:
            return preset
    # Fallback: return 'L' preset if present, else first preset
    return presets.get('L', next(iter(presets.values())))


def load_model_with_checkpoint(model_path, model_type, device='cpu'):
    """
    Load a trained model from checkpoint.
    For models without saved architecture, reconstruct from preset.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

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
        # Models that don't save architecture – reconstruct from preset file
        preset_file = Path(model_path).parent.parent / 'model_presets.json'

        with open(preset_file, 'r') as f:
            presets = json.load(f)

        preset = _find_preset_for_checkpoint(model_path, presets)

        posenc = preset.get('posenc', {'n_freqs': 4, 'scale': 1.0})

        # Base encoder config (2-D coordinates only)
        encoder_cfg = {
            'latent_dim': preset['latent_dim'],
            'pre_hidden': preset['pre_hidden'],
            'sa_blocks': preset['sa_blocks'],
            'gf_hidden': preset['gf_hidden'],
            'norm': preset.get('norm', 'batch'),
            'num_groups': preset.get('num_groups', 16),
            'pool': preset.get('pool', 'max'),
            'posenc': posenc,
        }

        if model_type in ('Pn2NoSDF', 'Pn2wSDF'):
            # ScaledDiagramDeepONet (PointNet++ branch, SIREN trunk)
            if model_type == 'Pn2wSDF':
                encoder_cfg['sdf_ch'] = 1  # encoder receives (x, y, sdf)
            basis_dim = int(preset.get('basis_dim', preset['head_hidden'][-1]))
            head_hidden = list(preset.get('post_mlp_hidden', preset['head_hidden']))
            siren_hidden = list(preset.get('siren_hidden', [256, 256]))
            model = ScaledDiagramDeepONet(
                latent_dim=preset['latent_dim'],
                basis_dim=basis_dim,
                head_hidden=head_hidden,
                siren_hidden=siren_hidden,
                encoder_cfg=encoder_cfg,
            )

        elif model_type == 'PointDeepONet':
            # PointDeepONet (VanillaPointNet branch, SIREN trunk, uses SDF)
            basis_dim = int(preset.get('basis_dim', preset['head_hidden'][-1]))
            head_hidden = preset['head_hidden']
            branch_hidden = list(
                preset.get(
                    'branch_hidden',
                    head_hidden[:-1] if len(head_hidden) > 1 else [64, 128, 256],
                )
            )
            siren_hidden = list(preset.get('siren_hidden', [256, 256]))
            post_mlp_hidden = list(preset.get('post_mlp_hidden', head_hidden))
            model = PointDeepONet(
                in_ch=3,  # x, y, sdf
                latent_dim=preset['latent_dim'],
                basis_dim=basis_dim,
                branch_hidden=branch_hidden,
                siren_hidden=siren_hidden,
                post_mlp_hidden=post_mlp_hidden,
                norm=preset.get('norm', 'batch'),
                num_groups=int(preset.get('num_groups', 16)),
            )

        # ---- Legacy model types (kept for backward compatibility) ----
        elif model_type in ('SpectralDeepONet', 'VanillaDeepONet', 'DenseNoFFT',
                            'Point_DeepONet'):
            head_posenc = preset.get('head_posenc', {'n_freqs': 4, 'scale': 1.0})
            head_hidden = preset['head_hidden']
            if len(head_hidden) > 1:
                do_branch_hidden = head_hidden[:-1]
                do_basis_dim = head_hidden[-1]
            else:
                do_branch_hidden = [256, 256]
                do_basis_dim = 128
            do_trunk_hidden = do_branch_hidden

            if model_type == 'SpectralDeepONet':
                n_freqs = head_posenc.get('n_freqs', 10)
                scale = head_posenc.get('scale', 2.0)
                model = SpectralDeepONet(
                    latent_dim=preset['latent_dim'],
                    basis_dim=do_basis_dim,
                    branch_hidden=do_branch_hidden,
                    trunk_hidden=do_trunk_hidden,
                    n_freqs=n_freqs,
                    scale=scale,
                    encoder_cfg=encoder_cfg,
                )
            elif model_type == 'VanillaDeepONet':
                model = VanillaDeepONet(
                    latent_dim=preset['latent_dim'],
                    basis_dim=do_basis_dim,
                    branch_hidden=do_branch_hidden,
                    trunk_hidden=do_trunk_hidden,
                    encoder_cfg=encoder_cfg,
                )
            elif model_type == 'DenseNoFFT':
                model = DenseNoFFT(
                    latent_dim=preset['latent_dim'],
                    mlp_hidden=head_hidden,
                    encoder_cfg=encoder_cfg,
                )
            else:  # Point_DeepONet (legacy)
                basis_dim = head_hidden[-1] if head_hidden else 128
                model = ScaledDiagramDeepONet(
                    latent_dim=preset['latent_dim'],
                    basis_dim=basis_dim,
                    head_hidden=head_hidden,
                    encoder_cfg=encoder_cfg,
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    # Extract normalization params (sdf_mean/sdf_std present only for SDF models)
    norm_params = {
        'coord_center': ckpt['coord_center'].to(device),
        'coord_half_range': ckpt['coord_half_range'].to(device),
        'stress_mean': ckpt['stress_mean'].to(device),
        'stress_std': ckpt['stress_std'].to(device),
    }
    if ckpt.get('sdf_mean') is not None:
        norm_params['sdf_mean'] = ckpt['sdf_mean'].to(device)
        norm_params['sdf_std'] = ckpt['sdf_std'].to(device)

    return model, norm_params, ckpt
