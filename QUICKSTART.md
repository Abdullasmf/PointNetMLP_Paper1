# Model Benchmarking Quick Reference

## Models Overview

| Model | Architecture | Special Features | File Location |
|-------|-------------|------------------|---------------|
| **SpectralDeepONet** | DeepONet with Fourier trunk | Spectral positional encoding | `SpectralDeepONet/benchmarks.py` |
| **DenseNoFFT** | Dense concatenation | No Fourier features | `DenseNoFFT/benchmarks.py` |
| **PointNetMLPJoint** | Joint encoder-decoder | Standard architecture | `PointNetMLPJoint/pn_models.py` |
| **VanillaDeepONet** | Standard DeepONet | Baseline without encoding | `VanillaDeepONet/benchmarks.py` |

## Standardized Configuration

All models use **identical** training settings:

### Training Parameters
```python
epochs = 500                     # Maximum epochs (with early stopping)
lr = 3e-4                        # Learning rate
weight_decay = 1e-4              # L2 regularization
batch_size = 8                   # Batch size
early_stopping_patience = 40     # Epochs without improvement
grad_clip_norm = 0.5             # Gradient clipping threshold
train_mode = "batched_all"       # Training mode
```

### Data Configuration
```python
data_file = "L_Bracket/L_bracket_stress.h5"
train_val_split = 0.2            # 80% train, 20% val
random_seed = 42                 # For reproducibility
preset = "L_full"                # Model architecture preset
```

### Model Architecture (L_full preset)
```python
latent_dim = 192
pre_hidden = [128, 128]
sa_blocks = [
    {"n_samples": 0, "radius": 0.04, "max_k": 32, "out_ch": 192, "mlp_hidden": [128]},
    {"n_samples": 0, "radius": 0.12, "max_k": 32, "out_ch": 384, "mlp_hidden": [256]}
]
gf_hidden = [128]
head_hidden = [512, 512, 256]
```

## Quick Start

### 1. Verify Data
```bash
# Check data file exists
ls -lh L_Bracket/L_bracket_stress.h5
```

### 2. Train Models
```bash
# Train all models sequentially
for model in SpectralDeepONet DenseNoFFT PointNetMLPJoint VanillaDeepONet; do
    echo "Training $model..."
    cd $model
    python Training_script.py
    cd ..
done
```

### 3. Monitor Training
Each model logs:
- Train MSE (normalized)
- Val MSE (normalized)
- Val MSE (MPa²) - actual stress units
- R² score (MPa space)
- Learning rate
- Epoch time

Example output:
```
Epoch 050 | train MSE: 0.045123 | val MSE: 0.052341 | val MSE(MPa^2): 8.234 | R2(MPa): 0.9234 | lr: 2.34e-04 | epoch: 42.1s
```

### 4. Find Best Models
```bash
# List saved checkpoints
ls -lh */Trained_models/*.pt
```

## File Structure Summary

```
Repository/
│
├── SpectralDeepONet/
│   ├── Training_script.py         ← Main entry point
│   ├── benchmarks.py               ← SpectralDeepONet class
│   ├── pn_models.py                ← Shared encoder
│   ├── model_presets.json          ← Architecture configs
│   └── Trained_models/             ← Checkpoints (created during training)
│
├── DenseNoFFT/
│   ├── Training_script.py         ← Main entry point
│   ├── benchmarks.py               ← DenseNoFFT class
│   ├── pn_models.py                ← Shared encoder
│   ├── model_presets.json          ← Architecture configs
│   └── Trained_models/             ← Checkpoints (created during training)
│
├── PointNetMLPJoint/
│   ├── Training_script.py         ← Main entry point
│   ├── pn_models.py                ← PointNetMLPJoint class + encoder
│   ├── model_presets.json          ← Architecture configs
│   └── Trained_models/             ← Checkpoints (created during training)
│
├── VanillaDeepONet/
│   ├── Training_script.py         ← Main entry point
│   ├── benchmarks.py               ← VanillaDeepONet class
│   ├── pn_models.py                ← Shared encoder
│   ├── model_presets.json          ← Architecture configs
│   └── Trained_models/             ← Checkpoints (created during training)
│
├── L_Bracket/
│   ├── L_bracket_stress.h5        ← Training data (HDF5)
│   └── generate_l_bracket.py      ← Data generation script
│
├── README.md                       ← Full documentation
├── TRAINING_VALIDATION.md          ← Validation report
└── .gitignore                      ← Ignore cache/data files
```

## Common Tasks

### Change Preset
Edit the `main()` call in Training_script.py:
```python
if __name__ == "__main__":
    main("L_full_ln_pos12", 8)  # Use different preset
```

Available presets: `L_full`, `L_full_ln_pos12`, `L_full_ln_pos12_drop`, etc.

### Change Batch Size
```python
if __name__ == "__main__":
    main("L_full", 16)  # Use batch size 16
```

### Resume Training
Not directly supported. To continue training:
1. Load checkpoint in Training_script.py
2. Initialize model from saved state
3. Continue training loop

### Evaluate Model
Load checkpoint and run validation:
```python
import torch
ckpt = torch.load("Trained_models/model.pt")
model.load_state_dict(ckpt["model_state"])
# Use ckpt["coord_center"], ckpt["stress_mean"], etc. for denormalization
```

## Troubleshooting

### Issue: "Preset 'S0' not found"
**Solution**: Use `"L_full"` preset (already fixed in all scripts)

### Issue: "STG1_edge.h5 not found"
**Solution**: Use `L_bracket_stress.h5` (already fixed in PointNetMLPJoint)

### Issue: Different results across runs
**Solution**: Check random seed is set to 42 in all scripts (already standardized)

### Issue: Out of memory
**Solution**: Reduce batch size:
```python
main("L_full", 4)  # Reduce from 8 to 4
```

## Verification Checklist

Before running experiments:
- [ ] Data file `L_Bracket/L_bracket_stress.h5` exists
- [ ] All Training_script.py files use `main("L_full", 8)`
- [ ] All scripts load from same data file
- [ ] Random seed is 42 in all scripts
- [ ] Git repo is clean (no uncommitted changes)

## Results Comparison

After training all models:
1. Compare validation MSE (MPa²) - lower is better
2. Compare R² scores - higher is better (max 1.0)
3. Compare training time per epoch
4. Compare total parameters (model size)

Example comparison:
```
Model              | Val MSE (MPa²) | R² Score | Params | Time/Epoch
-------------------|----------------|----------|--------|------------
SpectralDeepONet   | 7.234          | 0.9456   | 2.1M   | 45s
DenseNoFFT         | 8.123          | 0.9312   | 1.8M   | 38s
PointNetMLPJoint   | 7.891          | 0.9378   | 1.9M   | 41s
VanillaDeepONet    | 8.456          | 0.9289   | 2.0M   | 43s
```

## Contact

For questions or issues, please open an issue on GitHub.
