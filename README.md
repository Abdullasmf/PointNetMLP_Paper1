# PointNet-MLP Benchmarking Repository

This repository contains benchmarking code for 4 different neural network architectures for stress prediction on L-bracket geometries.

## Models

### 1. SpectralDeepONet
**Architecture**: DeepONet with Fourier features
- **Branch Network**: PointNet2Encoder2D → MLP → basis coefficients
- **Trunk Network**: Fourier features of coordinates → MLP → basis functions
- **Fusion**: Dot product of branch and trunk outputs
- **Key Feature**: Uses spectral/Fourier positional encoding for enhanced spatial resolution

### 2. DenseNoFFT
**Architecture**: Dense concatenation without Fourier features
- **Encoder**: PointNet2Encoder2D → latent representation
- **Decoder**: Concatenate [latent, query_coords] → MLP → stress prediction
- **Key Feature**: Direct concatenation of geometry encoding with query coordinates

### 3. PointNetMLPJoint
**Architecture**: Joint PointNet encoder with MLP head
- **Encoder**: PointNet2Encoder2D → latent representation
- **Decoder**: Concatenate [latent, query_coords] → MLP → stress prediction
- **Key Feature**: Standard joint architecture for point cloud regression

### 4. VanillaDeepONet
**Architecture**: Standard DeepONet without Fourier features
- **Branch Network**: PointNet2Encoder2D → MLP → basis coefficients
- **Trunk Network**: Raw coordinates → MLP → basis functions
- **Fusion**: Dot product of branch and trunk outputs
- **Key Feature**: Baseline DeepONet without positional encoding

## Repository Structure

```
.
├── SpectralDeepONet/
│   ├── Training_script.py      # Training script
│   ├── benchmarks.py            # Model definitions
│   ├── pn_models.py            # Shared encoder components
│   └── model_presets.json      # Architecture presets
├── DenseNoFFT/
│   ├── Training_script.py
│   ├── benchmarks.py
│   ├── pn_models.py
│   └── model_presets.json
├── PointNetMLPJoint/
│   ├── Training_script.py
│   ├── pn_models.py
│   └── model_presets.json
├── VanillaDeepONet/
│   ├── Training_script.py
│   ├── benchmarks.py
│   ├── pn_models.py
│   └── model_presets.json
├── L_Bracket/
│   ├── L_bracket_stress.h5     # Training data
│   └── generate_l_bracket.py   # Data generation script
└── TRAINING_VALIDATION.md      # Validation report
```

## Data

The training data is stored in `L_Bracket/L_bracket_stress.h5` with the following structure:
- **Format**: HDF5 file with groups `sample_0`, `sample_1`, ..., `sample_N`
- Each sample contains:
  - `points`: (N, 2) array of (x, y) coordinates
  - `stress`: (N, 1) array of stress values
  - `corner`: (2,) corner position for L-bracket geometry

## Training Configuration

All models use **standardized** training settings for fair comparison:

### Hyperparameters
- **Epochs**: 500 (with early stopping)
- **Learning Rate**: 3e-4
- **Weight Decay**: 1e-4
- **Batch Size**: 8
- **Optimizer**: AdamW
- **LR Scheduler**: OneCycleLR (cosine annealing)
- **Gradient Clipping**: 0.5
- **Early Stopping**: 40 epochs patience

### Data Processing
- **Train/Val Split**: 80/20 (random_state=42)
- **Normalization**: 
  - Coordinates: center and half-range normalization
  - Stress: mean and standard deviation normalization
- **Training Mode**: "batched_all" (uses all nodes with padding)

### Model Architecture (L_full preset)
- **Latent Dimension**: 192
- **Pre-hidden Layers**: [128, 128]
- **Set Abstraction Blocks**: 2
  - Block 1: radius=0.04, max_k=32, out_ch=192
  - Block 2: radius=0.12, max_k=32, out_ch=384
- **Global Feature Hidden**: [128]
- **Head Hidden Layers**: [512, 512, 256]

## Usage

### Requirements
```bash
pip install torch numpy h5py scikit-learn
```

### Training a Model

To train any model, navigate to its directory and run the training script:

```bash
# Train SpectralDeepONet
cd SpectralDeepONet
python Training_script.py

# Train DenseNoFFT
cd ../DenseNoFFT
python Training_script.py

# Train PointNetMLPJoint
cd ../PointNetMLPJoint
python Training_script.py

# Train VanillaDeepONet
cd ../VanillaDeepONet
python Training_script.py
```

### Output

Each training script will:
1. Load data from `../L_Bracket/L_bracket_stress.h5`
2. Split into train/validation sets (80/20)
3. Train for up to 500 epochs with early stopping
4. Log metrics: train MSE, validation MSE, validation MSE (MPa²), R² score
5. Save best model checkpoint to `Trained_models/` directory

Example output:
```
Starting training script with preset 'L_full' and batch size 8
Using device: cuda
Loading data from: /path/to/L_Bracket/L_bracket_stress.h5
Found 100 samples. Loading...
Loaded 100 datasets from the HDF5 file.
Coord center=[...], half_range=[...] | stress_mean=..., stress_std=...
Using 'batched_all' training with batch size 8
Saving best checkpoint to: Trained_models/pnmlp_abc12345.pt

Epoch 001 | train MSE: 0.123456 | val MSE: 0.234567 | val MSE(MPa^2): 12.345 | R2(MPa): 0.8912 | lr: 3.00e-05 | epoch: 45.2s
...
```

### Customization

To use a different preset or batch size, modify the `main()` call in the training script:

```python
if __name__ == "__main__":
    try:
        # Change preset (L_full, L_full_ln_pos12, etc.) or batch size
        main("L_full", batch=16)  # Use batch size 16
    except Exception as e:
        print(f"Error during training: {e}")
        raise
```

## Model Checkpoints

Trained models are saved with:
- Model state dictionary
- Architecture configuration
- Normalization parameters (coord_center, coord_half_range, stress_mean, stress_std)
- Training metrics (best validation loss, epochs trained)

Checkpoint naming: `{model_name}_{arch_hash}.pt`

## Validation

See `TRAINING_VALIDATION.md` for detailed validation of:
- Training script correctness
- Configuration standardization
- Data loading consistency
- Hyperparameter verification

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]
