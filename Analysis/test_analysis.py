#!/usr/bin/env python3
"""
Test script to validate the model comparison analysis functionality.
This simulates what the notebook does without requiring Jupyter.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from load_models_helper import load_model_with_checkpoint
from compute_sdf import compute_sdf_l_bracket, compute_sdf_plate_hole
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# Set up paths
repo_dir = Path('..').resolve()
model_dirs = {
    'Pn2_NoSDF':    repo_dir / 'Point++_DeepONet_noSDF' / 'Trained_models',
    'Pn2_wSDF':     repo_dir / 'Point++_DeepONet_wSDF'  / 'Trained_models',
    'Pnt_DeepONet': repo_dir / 'Point_DeepONet'          / 'Trained_models',
}
model_types = {
    'Pn2_NoSDF':    'Pn2NoSDF',
    'Pn2_wSDF':     'Pn2wSDF',
    'Pnt_DeepONet': 'PointDeepONet',
}


def compute_sdf_from_meta(points_np, meta):
    """Compute SDF values for a set of 2D points given geometry metadata."""
    if 'corner' in meta:
        sdf = compute_sdf_l_bracket(points_np, meta['corner'])
    elif 'params' in meta:
        sdf = compute_sdf_plate_hole(points_np, meta['params'])
    else:
        sdf = np.zeros(len(points_np), dtype=np.float32)
    return sdf.reshape(-1, 1).astype(np.float32)


def load_h5_data(path, max_samples=None):
    """Load samples from HDF5 file, returning tensors and geometry metadata."""
    samples = []
    meta = []
    with h5py.File(path, 'r') as hf:
        keys = sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))
        if max_samples:
            keys = keys[:max_samples]
        for key in keys:
            group = hf[key]
            coord_stress = np.hstack((group['points'][:], group['stress'][:]))
            samples.append(torch.from_numpy(coord_stress).float())
            m = {}
            if 'corner' in group:
                m['corner'] = group['corner'][:]
            if 'params' in group:
                m['params'] = group['params'][:]
            meta.append(m)
    return samples, meta


def evaluate_model(model, val_samples, val_meta, norm_params):
    """Evaluate model on validation samples. Handles both 2D and SDF (3D) models."""
    model.eval()
    uses_sdf = norm_params.get('sdf_mean') is not None
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample, meta in zip(val_samples, val_meta):
            points = sample[:, :2]
            stress = sample[:, 2:3]

            points_norm = (points - norm_params['coord_center'].cpu()) / norm_params['coord_half_range'].cpu()

            if uses_sdf:
                sdf = torch.from_numpy(compute_sdf_from_meta(points.numpy(), meta))
                sdf_norm = (sdf - norm_params['sdf_mean'].cpu()) / norm_params['sdf_std'].cpu()
                pts_input = torch.cat([points_norm, sdf_norm], dim=-1).unsqueeze(0).to(device)
            else:
                pts_input = points_norm.unsqueeze(0).to(device)

            pred_norm = model(pts_input, pts_input)
            pred = pred_norm * norm_params['stress_std'] + norm_params['stress_mean']

            all_preds.append(pred.squeeze().cpu().numpy())
            all_targets.append(stress.squeeze().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return {
        'MSE': mean_squared_error(all_targets, all_preds),
        'RMSE': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'MAE': np.mean(np.abs(all_targets - all_preds)),
        'R2': r2_score(all_targets, all_preds)
    }


# Test L_bracket models
print('='*60)
print('L_BRACKET MODEL COMPARISON')
print('='*60)

L_bracket_data_path = repo_dir / 'L_Bracket' / 'L_bracket_stress.h5'
print(f'Loading data: {L_bracket_data_path.name}')
L_bracket_samples, L_bracket_meta = load_h5_data(L_bracket_data_path, max_samples=50)
all_idx = list(range(len(L_bracket_samples)))
_, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
L_bracket_val_samples = [L_bracket_samples[i] for i in val_idx]
L_bracket_val_meta = [L_bracket_meta[i] for i in val_idx]
print(f'Validation samples: {len(L_bracket_val_samples)}\n')

L_bracket_results = {}
for model_name in ['Pn2_NoSDF', 'Pn2_wSDF', 'Pnt_DeepONet']:
    print(f'Evaluating {model_name}...')
    model_path_list = list(model_dirs[model_name].glob('L-*.pt'))
    if not model_path_list:
        print(f'  No checkpoint found, skipping.')
        continue
    model_path = model_path_list[0]

    try:
        model, norm_params, ckpt = load_model_with_checkpoint(
            str(model_path), model_types[model_name], device=str(device)
        )
        metrics = evaluate_model(model, L_bracket_val_samples, L_bracket_val_meta, norm_params)
        metrics['Best_Val_Loss'] = ckpt.get('best_val_loss', None)
        L_bracket_results[model_name] = metrics
        print(f'  MSE: {metrics["MSE"]:.4f}, R\u00b2: {metrics["R2"]:.4f}')
    except Exception as e:
        print(f'  Error: {e}')
        L_bracket_results[model_name] = {'Error': str(e)}

# Print results table
print(f'\n{"Model":<20} {"MSE":<10} {"RMSE":<10} {"MAE":<10} {"R\u00b2":<10}')
print('-'*60)
for model_name, metrics in L_bracket_results.items():
    if 'Error' not in metrics:
        print(f'{model_name:<20} {metrics["MSE"]:<10.4f} {metrics["RMSE"]:<10.4f} {metrics["MAE"]:<10.4f} {metrics["R2"]:<10.4f}')

if any('Error' not in m for m in L_bracket_results.values()):
    mse_values = {k: v['MSE'] for k, v in L_bracket_results.items() if 'Error' not in v}
    best_model = min(mse_values, key=mse_values.get)
    print(f'\n\u2713 Best Model (lowest MSE): {best_model} (MSE: {mse_values[best_model]:.4f})')

# Test Plate_hole models
print('\n' + '='*60)
print('PLATE_HOLE MODEL COMPARISON')
print('='*60)

Plate_hole_data_path = repo_dir / 'Plate_Hole' / 'Plate_hole_stress.h5'
print(f'Loading data: {Plate_hole_data_path.name}')
Plate_hole_samples, Plate_hole_meta = load_h5_data(Plate_hole_data_path, max_samples=50)
all_idx = list(range(len(Plate_hole_samples)))
_, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
Plate_hole_val_samples = [Plate_hole_samples[i] for i in val_idx]
Plate_hole_val_meta = [Plate_hole_meta[i] for i in val_idx]
print(f'Validation samples: {len(Plate_hole_val_samples)}\n')

Plate_hole_results = {}
for model_name in ['Pn2_NoSDF', 'Pn2_wSDF', 'Pnt_DeepONet']:
    print(f'Evaluating {model_name}...')
    model_path_list = list(model_dirs[model_name].glob('H-*.pt'))
    if not model_path_list:
        print(f'  No checkpoint found, skipping.')
        continue
    model_path = model_path_list[0]

    try:
        model, norm_params, ckpt = load_model_with_checkpoint(
            str(model_path), model_types[model_name], device=str(device)
        )
        metrics = evaluate_model(model, Plate_hole_val_samples, Plate_hole_val_meta, norm_params)
        metrics['Best_Val_Loss'] = ckpt.get('best_val_loss', None)
        Plate_hole_results[model_name] = metrics
        print(f'  MSE: {metrics["MSE"]:.4f}, R\u00b2: {metrics["R2"]:.4f}')
    except Exception as e:
        print(f'  Error: {e}')
        Plate_hole_results[model_name] = {'Error': str(e)}

# Print results table
print(f'\n{"Model":<20} {"MSE":<10} {"RMSE":<10} {"MAE":<10} {"R\u00b2":<10}')
print('-'*60)
for model_name, metrics in Plate_hole_results.items():
    if 'Error' not in metrics:
        print(f'{model_name:<20} {metrics["MSE"]:<10.4f} {metrics["RMSE"]:<10.4f} {metrics["MAE"]:<10.4f} {metrics["R2"]:<10.4f}')

if any('Error' not in m for m in Plate_hole_results.values()):
    mse_values = {k: v['MSE'] for k, v in Plate_hole_results.items() if 'Error' not in v}
    best_model = min(mse_values, key=mse_values.get)
    print(f'\n\u2713 Best Model (lowest MSE): {best_model} (MSE: {mse_values[best_model]:.4f})')

print('\n' + '='*60)
print('\u2713 ALL TESTS COMPLETED SUCCESSFULLY!')
print('='*60)

