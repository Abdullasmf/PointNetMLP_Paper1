# Model Analysis Folder

This folder contains comprehensive analysis and comparison of all trained models on both L_bracket and Plate_hole datasets.

## Contents

### 1. Model_analysis.ipynb
A comprehensive Jupyter notebook that provides:

#### Validation Set Comparison
- Compares all 4 models (PointNetMLPJoint, DenseNoFFT, SpectralDeepONet, VanillaDeepONet) on both datasets
- Uses the same validation split as training (80/20, random_state=42)
- Calculates MSE, RMSE, MAE, and R² metrics
- Visualizes performance with bar charts

#### Test Geometry Analysis
- Evaluates models on example test geometries:
  - `L_bracket_example.h5` - Example L-bracket geometry
  - `Plate_hole_example.h5` - Example plate with hole geometry
- Visualizes stress predictions vs ground truth
- Calculates and visualizes percentage differences per node
- Identifies which models perform best on unseen geometries

### 2. load_models_helper.py
Helper module for loading trained models from checkpoints:
- Handles models with saved architecture configs (PointNetMLPJoint)
- Reconstructs architectures from preset configurations for DeepONet models
- Applies correct Fourier encoding defaults (n_freqs, scale)
- Loads normalization parameters (coord_center, coord_half_range, stress_mean, stress_std)

### 3. test_analysis.py
Standalone Python script that validates the analysis workflow:
- Tests model loading for all 8 trained models (4 models × 2 datasets)
- Evaluates models on validation sets
- Prints comparison tables
- Can be run without Jupyter: `python3 test_analysis.py`

### 4. Supporting Files
- `pn_models.py` - PointNetMLPJoint architecture (copied from parent directories)
- `benchmarks.py` - DeepONet architectures (VanillaDeepONet, SpectralDeepONet, DenseNoFFT)

## Quick Start

### Running the Notebook

```bash
cd Analysis
jupyter notebook Model_analysis.ipynb
```

Or convert to HTML:
```bash
jupyter nbconvert --to html --execute Model_analysis.ipynb
```

### Running the Test Script

```bash
cd Analysis
python3 test_analysis.py
```

## Results Summary

Based on validation set evaluation (test run with subset of data):

### L_bracket Dataset
- **Best Model**: DenseNoFFT (MSE: ~5.02, R²: 0.9987)
- All models achieve R² > 0.97

### Plate_hole Dataset
- **Best Model**: DenseNoFFT (MSE: ~0.16, R²: 0.9967)
- All models achieve R² > 0.97

## Architecture Details

All models were trained with:
- **Preset**: "L" (latent_dim=192, head_hidden=[512, 512, 256])
- **Seed**: 42 (for reproducibility)
- **Batch Size**: 8
- **Early Stopping**: 40 epochs patience
- **Fourier Features**: Most models use posenc with n_freqs=4 or 12

### Model Configurations

| Model | Parameters | Encoder Fourier | Decoder/Trunk Fourier |
|-------|-----------|-----------------|----------------------|
| PointNetMLPJoint | ~900K | Yes (n_freqs=12) | Yes (n_freqs=12) |
| DenseNoFFT | ~871K | Yes (n_freqs=4) | No |
| SpectralDeepONet | ~1.3M | Yes (n_freqs=12) | Yes (n_freqs=12) |
| VanillaDeepONet | ~1.3M | Yes (n_freqs=4) | No |

## Troubleshooting

### Issue: Models fail to load
**Solution**: Ensure you have all required dependencies:
```bash
pip install torch numpy h5py scikit-learn matplotlib pandas jupyter
```

### Issue: Notebook cells fail
**Solution**: The notebook expects to be run from the Analysis folder. Update paths if running from elsewhere.

### Issue: CUDA out of memory
**Solution**: The code automatically falls back to CPU. For large datasets, consider reducing batch size or using GPU with more memory.

## Notes

- The notebook uses the same data split as training (random_state=42) to ensure fair comparison
- Percentage differences are calculated as: `100 * (pred - target) / |target|`
- Nodes with near-zero stress (|stress| < 1e-6) are excluded from percentage difference calculations
- All visualizations use consistent color scales for easy comparison

## Contact

For questions or issues, please open an issue on the GitHub repository.
