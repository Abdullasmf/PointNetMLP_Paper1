# Training Script and Model Configuration Validation Report

This document validates that all 4 models in the benchmarking repository have correctly configured and standardized training scripts and model presets.

## Models Benchmarked
1. **SpectralDeepONet** - DeepONet with Spectral/Fourier features in trunk network
2. **DenseNoFFT** - Dense concatenation without FFT
3. **PointNetMLPJoint** - PointNet encoder with MLP head (joint architecture)
4. **VanillaDeepONet** - Standard DeepONet without Fourier features

## Training Script Standardization

### ✅ Data Loading
All models now load data from the same source:
- **Data File**: `L_Bracket/L_bracket_stress.h5`
- **Data Format**: HDF5 with keys `sample_0`, `sample_1`, etc.
- **Data Structure**: Each sample contains:
  - `points`: (N, 2) coordinate array
  - `stress`: (N, 1) stress values
  - `corner`: (2,) corner position
- **Loading Function**: All use consistent `load_h5_pointsets()` implementation

### ✅ Training Hyperparameters
All models use identical training hyperparameters:
- **Epochs**: 500
- **Learning Rate**: 3e-4
- **Weight Decay**: 1e-4
- **Early Stopping Patience**: 40 epochs
- **Gradient Clipping**: 0.5
- **Training Mode**: "batched_all" (uses all nodes per geometry with padding)
- **Batch Size**: 8 (configurable via main function parameter)
- **Optimizer**: AdamW
- **LR Scheduler**: OneCycleLR with:
  - pct_start: 0.1
  - div_factor: 10.0
  - final_div_factor: 10.0
  - anneal_strategy: "cos"

### ✅ Data Preprocessing
All models use consistent normalization:
- **Coordinate Normalization**: Center and half-range normalization
- **Stress Normalization**: Mean and standard deviation normalization
- **Train/Val Split**: 80/20 split with random_state=42
- **Random Seed**: 42 (for reproducibility)

### ✅ Model Preset Configuration
All models use the "L_full" preset with identical architecture parameters:
- **Latent Dimension**: 192
- **Pre-hidden Layers**: [128, 128]
- **Set Abstraction Blocks**: 2 blocks
  - Block 1: radius=0.04, max_k=32, out_ch=192
  - Block 2: radius=0.12, max_k=32, out_ch=384
- **Global Feature Hidden**: [128]
- **Head Hidden Layers**: [512, 512, 256]
- **Normalization**: Batch normalization
- **Pooling**: Max pooling

### ✅ Training Process
All models follow the same training workflow:
1. Load and preprocess data
2. Compute global normalization statistics from training set
3. Create geometry-level datasets with normalization
4. Use AllNodesPadCollate for batching (handles variable point counts)
5. Train with identical loss function (MSE)
6. Validate on held-out set
7. Track both normalized and MPa-space metrics
8. Save best model checkpoint with:
   - Model state dict
   - Architecture configuration
   - Normalization parameters
   - Training metrics

### ✅ Model Architecture Differences
While training is standardized, each model has a distinct architecture:

1. **SpectralDeepONet**:
   - Branch: PointNet2Encoder2D → MLP
   - Trunk: Fourier features → MLP
   - Fusion: Dot product

2. **DenseNoFFT**:
   - Encoder: PointNet2Encoder2D
   - Decoder: Concatenate latent + query coords → MLP
   - No Fourier features

3. **PointNetMLPJoint**:
   - Encoder: PointNet2Encoder2D
   - Decoder: Concatenate latent + query coords → MLP
   - Standard joint architecture

4. **VanillaDeepONet**:
   - Branch: PointNet2Encoder2D → MLP
   - Trunk: Raw coords → MLP (no Fourier)
   - Fusion: Dot product

## Validation Checklist

### Configuration Files
- [x] All models have `model_presets.json` with "L_full" preset
- [x] Preset configurations are identical across all models
- [x] All presets include required fields (latent_dim, pre_hidden, sa_blocks, etc.)

### Training Scripts
- [x] All scripts call `main("L_full", 8)` (corrected from non-existent "S0")
- [x] All scripts load from same data file path
- [x] All scripts use identical hyperparameters
- [x] All scripts use same random seed (42)
- [x] All scripts use same train/val split (0.2)
- [x] All scripts use same training mode ("batched_all")
- [x] All scripts save models to "Trained_models" directory

### Data Processing
- [x] All use consistent data loading function
- [x] All apply same normalization strategy
- [x] All handle variable point counts consistently
- [x] All use same collate functions

### Model Instantiation
- [x] SpectralDeepONet correctly instantiated with Fourier features
- [x] DenseNoFFT correctly instantiated without Fourier features
- [x] PointNetMLPJoint correctly instantiated with joint architecture
- [x] VanillaDeepONet correctly instantiated without Fourier features
- [x] All models use shared PointNet2Encoder2D encoder

### Output and Logging
- [x] All models log identical metrics (train MSE, val MSE, val MSE(MPa²), R2)
- [x] All models save checkpoints with same structure
- [x] All models use consistent naming scheme (model_name + arch_hash)

## Summary

✅ **All training scripts are correctly configured and standardized.**

The repository is set up for fair benchmarking with:
- Identical training procedures
- Same data source and preprocessing
- Standardized hyperparameters
- Consistent evaluation metrics
- Only architectural differences between models

## Usage

To train each model:

```bash
# SpectralDeepONet
cd SpectralDeepONet
python Training_script.py

# DenseNoFFT
cd DenseNoFFT
python Training_script.py

# PointNetMLPJoint
cd PointNetMLPJoint
python Training_script.py

# VanillaDeepONet
cd VanillaDeepONet
python Training_script.py
```

All scripts will:
- Use preset "L_full" with batch size 8
- Load data from `../L_Bracket/L_bracket_stress.h5`
- Train for up to 500 epochs with early stopping
- Save best model to `Trained_models/` directory
