# PointNet-MLP Benchmarking Repository

This repository contains benchmarking code for 4 different neural network architectures for stress prediction on two different geometries: **L-bracket** and **Hole plate**.

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

The repository includes training data for two different geometries:

### L-Bracket Dataset
- **Location**: `L_Bracket/L_bracket_stress.h5`
- **Format**: HDF5 file with groups `sample_0`, `sample_1`, ..., `sample_N`
- Each sample contains:
  - `points`: (N, 2) array of (x, y) coordinates
  - `stress`: (N, 1) array of stress values
  - `corner`: (2,) corner position for L-bracket geometry

### Hole Plate Dataset
- **Location**: `Plate_Hole/Plate_hole_stress.h5`
- **Format**: HDF5 file with groups `sample_0`, `sample_1`, ..., `sample_N`
- Each sample contains:
  - `points`: (N, 2) array of (x, y) coordinates
  - `stress`: (N, 1) array of stress values
  - `params`: Geometry parameters for hole plate

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
- **Mixed Precision Training**: Enabled (AMP) with optimized scheduler step ordering

### Training Optimizations

The training scripts use several optimizations for efficient and stable training:

1. **Mixed Precision Training (AMP)**: Uses PyTorch's automatic mixed precision to reduce memory usage and speed up training on GPUs with tensor cores.

2. **Optimized Scheduler Ordering**: The learning rate scheduler is called in the correct order with gradient scaling:
   ```python
   scaler.step(optimizer)      # Step 1: Conditionally update weights (if gradients are finite)
   scheduler.step()             # Step 2: Update learning rate
   scaler.update()              # Step 3: Update gradient scaler
   ```
   This ordering prevents the PyTorch warning about calling `lr_scheduler.step()` before `optimizer.step()` and ensures proper learning rate tracking with AMP.

3. **Gradient Clipping**: Applied before optimizer step to prevent gradient explosion.

4. **OneCycleLR Scheduler**: Steps after each batch (not epoch) for smooth learning rate annealing throughout training.

### Data Processing
- **Train/Val Split**: 80/20 (random_state=42)
- **Normalization**: 
  - Coordinates: center and half-range normalization
  - Stress: mean and standard deviation normalization
- **Training Mode**: "batched_all" (uses all nodes with padding)

### Model Architecture Presets

Available presets in `model_presets.json` (S, M, L, XL, XXL):
- **S (Small)**: Latent dim 128, suitable for quick experiments
- **M (Medium)**: Latent dim 160, balanced performance
- **L (Large)**: Latent dim 192, recommended for benchmarking
- **XL (Extra Large)**: Latent dim 256, high capacity
- **XXL (Double Extra Large)**: Latent dim 384, maximum capacity

Example L preset architecture:
- **Latent Dimension**: 192
- **Pre-hidden Layers**: [128, 128]
- **Set Abstraction Blocks**: 2
  - Block 1: radius=0.02, max_k=48, out_ch=256
  - Block 2: radius=0.05, max_k=48, out_ch=512
- **Global Feature Hidden**: [128]
- **Head Hidden Layers**: [512, 512, 256]

## Usage

### Requirements
```bash
pip install torch numpy h5py scikit-learn
```

### Training a Model

All training scripts support both L-bracket and Hole plate datasets. To train any model, navigate to its directory and run the training script:

```bash
# Train SpectralDeepONet (default: L-bracket dataset)
cd SpectralDeepONet
python Training_script.py

# Train DenseNoFFT (default: L-bracket dataset)
cd ../DenseNoFFT
python Training_script.py

# Train PointNetMLPJoint (default: L-bracket dataset)
cd ../PointNetMLPJoint
python Training_script.py

# Train VanillaDeepONet (default: L-bracket dataset)
cd ../VanillaDeepONet
python Training_script.py
```

### Output

Each training script will:
1. Load data from the selected dataset (default: `L_Bracket/L_bracket_stress.h5`)
2. Split into train/validation sets (80/20)
3. Train for up to 500 epochs with early stopping
4. Log metrics: train MSE, validation MSE, validation MSE (MPa²), R² score
5. Save best model checkpoint to `Trained_models/` directory with geometry prefix

Example output:
```
Starting training script with preset 'L', batch size 8, and dataset 'L_bracket'
Using device: cuda
Loading data from: /path/to/L_Bracket/L_bracket_stress.h5
Found 2000 samples. Loading...
Loaded 2000 datasets from the HDF5 file.
Coord center=[...], half_range=[...] | stress_mean=..., stress_std=...
Using 'batched_all' training with batch size 8
Saving best checkpoint to: Trained_models/L-pnmlp_abc12345.pt

Epoch 001 | train MSE: 0.123456 | val MSE: 0.234567 | val MSE(MPa^2): 12.345 | R2(MPa): 0.8912 | lr: 3.00e-05 | epoch: 45.2s
...
```

### Customization

#### Selecting a Dataset

To train on a different dataset, modify the `main()` call in the training script:

```python
if __name__ == "__main__":
    try:
        # Train on L-bracket dataset (default)
        main("L", batch=8, dataset="L_bracket")
        
        # Or train on Hole plate dataset
        # main("L", batch=8, dataset="Plate_hole")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
```

**Dataset Options:**
- `"L_bracket"`: Use L-bracket geometry data (prefix: `L-`)
- `"Plate_hole"`: Use Hole plate geometry data (prefix: `H-`)

The saved model files will be automatically prefixed based on the dataset:
- L-bracket models: `L-pnmlp_abc12345.pt`
- Hole plate models: `H-pnmlp_abc12345.pt`

#### Changing Model Size or Batch Size

To use a different preset or batch size:

```python
if __name__ == "__main__":
    try:
        # Available presets: "S", "M", "L", "XL", "XXL"
        # Larger models need smaller batch sizes
        main("XL", batch=4, dataset="L_bracket")  # Use XL model with batch size 4
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

Checkpoint naming: `{geometry_prefix}{model_name}_{arch_hash}.pt`
- Examples: `L-pnmlp_abc12345.pt`, `H-spectral_l_def67890.pt`

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
