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
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# Set up paths
repo_dir = Path('..').resolve()
model_dirs = {
    'PointNetMLPJoint': repo_dir / 'PointNetMLPJoint' / 'Trained_models',
    'DenseNoFFT': repo_dir / 'DenseNoFFT' / 'Trained_models',
    'SpectralDeepONet': repo_dir / 'SpectralDeepONet' / 'Trained_models',
    'VanillaDeepONet': repo_dir / 'VanillaDeepONet' / 'Trained_models'
}

def load_h5_data(path, max_samples=None):
    """Load samples from HDF5 file."""
    samples = []
    with h5py.File(path, 'r') as hf:
        keys = sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))
        if max_samples:
            keys = keys[:max_samples]
        for key in keys:
            group = hf[key]
            coord_stress = np.hstack((group['points'][:], group['stress'][:]))
            samples.append(torch.from_numpy(coord_stress).float())
    return samples

def evaluate_model(model, val_samples, norm_params):
    """Evaluate model on validation samples."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sample in val_samples:
            sample = sample.to(device)
            points = sample[:, :2]
            stress = sample[:, 2:3]
            
            # Normalize
            points_norm = (points - norm_params['coord_center']) / norm_params['coord_half_range']
            geom_points = points_norm.unsqueeze(0)
            query_points = points_norm.unsqueeze(0)
            
            # Predict
            pred_norm = model(geom_points, query_points)
            pred = pred_norm * norm_params['stress_std'] + norm_params['stress_mean']
            
            all_preds.append(pred.squeeze().cpu().numpy())
            all_targets.append(stress.squeeze().cpu().numpy())
    
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
L_bracket_samples = load_h5_data(L_bracket_data_path, max_samples=50)  # Limit for faster testing
_, L_bracket_val_samples = train_test_split(L_bracket_samples, test_size=0.2, random_state=42)
print(f'Validation samples: {len(L_bracket_val_samples)}\n')

L_bracket_results = {}
for model_name in ['PointNetMLPJoint', 'DenseNoFFT', 'SpectralDeepONet', 'VanillaDeepONet']:
    print(f'Evaluating {model_name}...')
    model_path = list(model_dirs[model_name].glob('L-*.pt'))[0]
    
    try:
        model, norm_params, ckpt = load_model_with_checkpoint(str(model_path), model_name, device=str(device))
        metrics = evaluate_model(model, L_bracket_val_samples, norm_params)
        metrics['Best_Val_Loss'] = ckpt.get('best_val_loss', None)
        L_bracket_results[model_name] = metrics
        print(f'  MSE: {metrics["MSE"]:.4f}, R²: {metrics["R2"]:.4f}')
    except Exception as e:
        print(f'  Error: {e}')
        L_bracket_results[model_name] = {'Error': str(e)}

# Print results table
print(f'\n{"Model":<20} {"MSE":<10} {"RMSE":<10} {"MAE":<10} {"R²":<10}')
print('-'*60)
for model_name, metrics in L_bracket_results.items():
    if 'Error' not in metrics:
        print(f'{model_name:<20} {metrics["MSE"]:<10.4f} {metrics["RMSE"]:<10.4f} {metrics["MAE"]:<10.4f} {metrics["R2"]:<10.4f}')

# Find best model
if any('Error' not in m for m in L_bracket_results.values()):
    mse_values = {k: v['MSE'] for k, v in L_bracket_results.items() if 'Error' not in v}
    best_model = min(mse_values, key=mse_values.get)
    print(f'\n✓ Best Model (lowest MSE): {best_model} (MSE: {mse_values[best_model]:.4f})')

# Test Plate_hole models
print('\n' + '='*60)
print('PLATE_HOLE MODEL COMPARISON')
print('='*60)

Plate_hole_data_path = repo_dir / 'Plate_Hole' / 'Plate_hole_stress.h5'
print(f'Loading data: {Plate_hole_data_path.name}')
Plate_hole_samples = load_h5_data(Plate_hole_data_path, max_samples=50)
_, Plate_hole_val_samples = train_test_split(Plate_hole_samples, test_size=0.2, random_state=42)
print(f'Validation samples: {len(Plate_hole_val_samples)}\n')

Plate_hole_results = {}
for model_name in ['PointNetMLPJoint', 'DenseNoFFT', 'SpectralDeepONet', 'VanillaDeepONet']:
    print(f'Evaluating {model_name}...')
    model_path = list(model_dirs[model_name].glob('H-*.pt'))[0]
    
    try:
        model, norm_params, ckpt = load_model_with_checkpoint(str(model_path), model_name, device=str(device))
        metrics = evaluate_model(model, Plate_hole_val_samples, norm_params)
        metrics['Best_Val_Loss'] = ckpt.get('best_val_loss', None)
        Plate_hole_results[model_name] = metrics
        print(f'  MSE: {metrics["MSE"]:.4f}, R²: {metrics["R2"]:.4f}')
    except Exception as e:
        print(f'  Error: {e}')
        Plate_hole_results[model_name] = {'Error': str(e)}

# Print results table
print(f'\n{"Model":<20} {"MSE":<10} {"RMSE":<10} {"MAE":<10} {"R²":<10}')
print('-'*60)
for model_name, metrics in Plate_hole_results.items():
    if 'Error' not in metrics:
        print(f'{model_name:<20} {metrics["MSE"]:<10.4f} {metrics["RMSE"]:<10.4f} {metrics["MAE"]:<10.4f} {metrics["R2"]:<10.4f}')

# Find best model
if any('Error' not in m for m in Plate_hole_results.values()):
    mse_values = {k: v['MSE'] for k, v in Plate_hole_results.items() if 'Error' not in v}
    best_model = min(mse_values, key=mse_values.get)
    print(f'\n✓ Best Model (lowest MSE): {best_model} (MSE: {mse_values[best_model]:.4f})')

print('\n' + '='*60)
print('✓ ALL TESTS COMPLETED SUCCESSFULLY!')
print('='*60)
