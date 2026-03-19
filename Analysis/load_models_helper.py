import re
import sys
import importlib.util
import torch
import json
from pathlib import Path
from pn_models import PointNetMLPJoint
from benchmarks import (
    VanillaDeepONet, SpectralDeepONet, DenseNoFFT,
    ScaledDiagramDeepONet, PointDeepONet, ScaledDiagramDeepONetFFMCAtt,
)


def _import_class_from_folder(folder_path, class_name):
    """Dynamically import a class from the benchmarks.py in a training folder.

    This ensures that GINOT and ArGEnT model classes are loaded from the exact
    same source files used during training, rather than potentially diverged
    copies in the Analysis folder.
    """
    folder_path = Path(folder_path).resolve()
    benchmarks_file = folder_path / 'benchmarks.py'

    if not benchmarks_file.exists():
        raise FileNotFoundError(
            f"Cannot find benchmarks.py in training folder: {folder_path}. "
            "Expected path: " + str(benchmarks_file)
        )

    # Use a unique module name to avoid collisions with Analysis/benchmarks
    module_name = f'_benchmarks_{folder_path.name}'

    if module_name in sys.modules:
        return getattr(sys.modules[module_name], class_name)

    spec = importlib.util.spec_from_file_location(module_name, str(benchmarks_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # Add the training folder to sys.path so its local imports (e.g. pn_models)
    # resolve correctly.  The entry is kept for the lifetime of the process so
    # that any deferred imports inside the loaded module continue to work.
    folder_str = str(folder_path)
    if folder_str not in sys.path:
        sys.path.insert(0, folder_str)

    spec.loader.exec_module(module)

    return getattr(module, class_name)


def _parse_n_s(value, default=128):
    """Parse the n_s GINOT parameter, preserving float ratios in (0, 1]."""
    if value is None:
        return default
    if isinstance(value, float) and 0.0 < value <= 1.0:
        return value
    return int(value)


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

#model parameters count
def count_parameters(model_or_path):
    """Count parameters in a model or model checkpoint file.

    Accepts either:
    - a `torch.nn.Module` instance (counts trainable + non-trainable parameters
      via `p.numel()` for each parameter tensor), or
    - a file path to a `.pt` checkpoint (loads the state dict and sums tensor sizes).
    """
    import torch.nn as nn
    if isinstance(model_or_path, nn.Module):
        return sum(p.numel() for p in model_or_path.parameters())

    # Treat as a file path to a checkpoint
    model_path = model_or_path
    try:
        # PyTorch >=2.6 may default to weights_only=True and fail on full checkpoints
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Support common checkpoint key names
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError('Loaded state is not a parameter dictionary.')

        # Count only tensor entries (skip optimizer/scheduler/etc.)
        total_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
        return total_params
    except Exception as e:
        print(f'Error counting parameters for {model_path}: {e}')
        return None
    


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

        # training_folder is the parent of Trained_models/ – it contains benchmarks.py
        training_folder = Path(model_path).parent.parent

        # ---- ArGEnT models – import class from the training folder ----
        if model_type in ('ArGEnTCrossWSD', 'ArGEnTSelfNoSDF'):
            ArGEnTDeepONet = _import_class_from_folder(training_folder, 'ArGEnTDeepONet')
            if arch is not None:
                # Prefer arch dict saved in checkpoint (most accurate)
                model = ArGEnTDeepONet(
                    hidden_dim=int(arch.get("hidden_dim", 128)),
                    num_heads=int(arch.get("num_heads", 4)),
                    num_layers=int(arch.get("num_layers", 2)),
                    output_dim=int(arch.get("output_dim", 128)),
                    attention_type=arch.get("attention_type", "cross"),
                    use_sdf=bool(arch.get("use_sdf", model_type == 'ArGEnTCrossWSD')),
                )
            else:
                # Fallback: reconstruct from preset
                attention_type = "cross" if model_type == 'ArGEnTCrossWSD' else "self"
                use_sdf = (model_type == 'ArGEnTCrossWSD')
                model = ArGEnTDeepONet(
                    hidden_dim=int(preset.get("hidden_dim", 128)),
                    num_heads=int(preset.get("num_heads", 4)),
                    num_layers=int(preset.get("num_layers", 2)),
                    output_dim=int(preset.get("output_dim", 128)),
                    attention_type=attention_type,
                    use_sdf=use_sdf,
                )

        # ---- GINOT model – import class from the training folder ----
        elif model_type == 'GINOT_noSDF':
            GINOT = _import_class_from_folder(training_folder, 'GINOT')
            if arch is not None:
                # Use arch dict saved in checkpoint (most accurate)
                model = GINOT(
                    d_model=int(arch.get("d_model", 128)),
                    num_encoder_cross_layers=int(arch.get("num_encoder_cross_layers", 2)),
                    num_encoder_self_layers=int(arch.get("num_encoder_self_layers", 2)),
                    num_decoder_layers=int(arch.get("num_decoder_layers", 2)),
                    n_heads=int(arch.get("n_heads", 4)),
                    n_s=_parse_n_s(arch.get("n_s", 128)),
                    n_p=int(arch.get("n_p", 32)),
                    radius=float(arch.get("radius", 0.15)),
                    mlp_hidden_dims=list(arch.get("mlp_hidden_dims", [256, 256])),
                )
            else:
                # Fallback: reconstruct from preset
                d_model = int(preset.get("d_model", 128))
                model = GINOT(
                    d_model=d_model,
                    num_encoder_cross_layers=int(preset.get("num_encoder_cross_layers", 2)),
                    num_encoder_self_layers=int(preset.get("num_encoder_self_layers", 2)),
                    num_decoder_layers=int(preset.get("num_decoder_layers", 2)),
                    n_heads=int(preset.get("n_heads", 4)),
                    n_s=_parse_n_s(preset.get("n_s", 128)),
                    n_p=int(preset.get("n_p", 32)),
                    radius=float(preset.get("radius", 0.15)),
                    mlp_hidden_dims=list(preset.get("mlp_hidden_dims", [d_model * 2, d_model * 2])),
                )

        else:
            # All other models use PointNet++ encoder config from preset
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

            elif model_type in ('Pn2NoSDF_FFM_CAtt', 'Pn2wSDF_FFM_CAtt'):
                # ScaledDiagramDeepONetFFMCAtt (PointNet++ branch, FFM trunk, cross-attention)
                if model_type == 'Pn2wSDF_FFM_CAtt':
                    encoder_cfg['sdf_ch'] = 1  # encoder receives (x, y, sdf)
                basis_dim = int(preset.get('basis_dim', preset['head_hidden'][-1]))
                head_hidden = list(preset.get('head_hidden', [256, 256, 128]))
                ffm_mapping_size = int(preset.get('ffm_mapping_size', 128))
                ffm_sigma_init = float(preset.get('ffm_sigma_init', 2.0))
                cross_attention_heads = int(preset.get('cross_attention_heads', 4))
                attn_temp = float(preset.get('attn_temp', 0.1))
                model = ScaledDiagramDeepONetFFMCAtt(
                    latent_dim=preset['latent_dim'],
                    basis_dim=basis_dim,
                    head_hidden=head_hidden,
                    ffm_mapping_size=ffm_mapping_size,
                    ffm_sigma_init=ffm_sigma_init,
                    cross_attention_heads=cross_attention_heads,
                    attn_temp=attn_temp,
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

            elif model_type == 'PointDeepONetNoSDF':
                # PointDeepONet without SDF (VanillaPointNet branch, SIREN trunk, x,y only)
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
                    in_ch=2,  # x, y only — no SDF
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
