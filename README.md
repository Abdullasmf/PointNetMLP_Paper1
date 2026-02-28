# Point-DeepONet Benchmarking Repository

Benchmarking of three **Point-DeepONet** variants for stress-field prediction on 2-D FEM geometries. Each variant is a faithful ablation study: same training procedure, same **M-preset** size budget, different encoder/SDF configuration.

---

## Models

| Folder | Encoder | SDF in trunk? | Description |
|---|---|---|---|
| `Point_DeepONet` | Vanilla PointNet (global max-pool) | ✅ (x, y, sdf) | Park et al. baseline — no boundary conditions, SDF used as geometric feature |
| `Point++_DeepONet_noSDF` | PointNet++ (set abstraction) | ❌ (x, y only) | PointNet++ branch, (x, y)-only SIREN trunk |
| `Point++_DeepONet_wSDF` | PointNet++ (set abstraction) | ✅ (x, y, sdf) | PointNet++ branch, SDF-augmented SIREN trunk |

### Architecture overview

All three models follow the same **DeepONet diagram**:

```
Geometry point cloud
        │
   [ Branch encoder ]  ──►  B^α  [B, latent_dim]
                               │
                           element-wise multiply (⊙)
                               │
Query coordinates (± SDF)      │
        │                      │
   [ Trunk SIREN ]  ──────────► T^α  [B, Q, latent_dim]
                               │
                   [ Post-fusion MLP ]  ──►  B^β  [B, Q, basis_dim]
                               │
              T^β = Linear(T^α)  [B, Q, basis_dim]
                               │
               dot-product Σ(B^β · T^β) + bias  ──►  stress [B, Q, 1]
```

**Branch differences:**
- `Point_DeepONet` — `VanillaPointNetEncoder`: shared pointwise MLP → global max-pool. Input: (x, y, sdf).
- `Point++_DeepONet_*` — `PointNet2Encoder2D`: two set-abstraction layers → global feature. Input: (x, y) only (the PointNet++ branch is geometry-only in both noSDF and wSDF).

**Trunk differences:**
- `noSDF`: SIREN input = (x, y) — 2 channels.
- `wSDF` and `Point_DeepONet`: SIREN input = (x, y, sdf) — 3 channels.

---

## Repository Structure

```
.
├── Point_DeepONet/
│   ├── Training_script.py      # Data loading, training loop (SDF pipeline)
│   ├── benchmarks.py           # PointDeepONet, VanillaDeepONet, ScaledDiagramDeepONet
│   ├── pn_models.py            # VanillaPointNetEncoder, PointNet2Encoder2D, MLP
│   ├── model_presets.json      # S / M / L / XL / XXL preset definitions
│   └── GPU0.py                 # Entry point: runs preset M on both datasets
│
├── Point++_DeepONet_noSDF/
│   ├── Training_script.py      # (x, y)-only data pipeline
│   ├── benchmarks.py           # ScaledDiagramDeepONet (no-SDF variant)
│   ├── pn_models.py            # PointNet2Encoder2D with sdf_ch support
│   ├── model_presets.json
│   └── GPU0.py
│
├── Point++_DeepONet_wSDF/
│   ├── Training_script.py      # (x, y, sdf) data pipeline; sdf_ch=1 in encoder_cfg
│   ├── benchmarks.py           # ScaledDiagramDeepONet (SDF-in-trunk variant)
│   ├── pn_models.py            # PointNet2Encoder2D with sdf_ch support
│   ├── model_presets.json
│   └── GPU0.py
│
├── L_Bracket/
│   ├── L_bracket_stress.h5     # Training data — L-bracket FEM simulations
│   └── generate_l_bracket.py   # Data generation script
│
├── Plate_Hole/
│   ├── Plate_hole_stress.h5    # Training data — plate-with-hole FEM simulations
│   └── Generate_plate_hole.py  # Data generation script
│
├── Analysis/                   # Post-training analysis notebooks (not yet active)
│
├── compute_sdf.py              # Utility: compute and append SDF to HDF5 datasets
├── run0.sh                     # SLURM: Point_DeepONet (GPU 0)
├── run1.sh                     # SLURM: Point++_DeepONet_wSDF (GPU 1)
├── run2.sh                     # SLURM: Point++_DeepONet_noSDF (GPU 2)
├── QUICKSTART.md
└── TRAINING_VALIDATION.md
```

---

## Datasets

Both datasets are stored as **HDF5** files. Each sample is a group `sample_0`, `sample_1`, … containing:

| Key | Shape | Description |
|---|---|---|
| `points` | (N, 2) | (x, y) mesh node coordinates (normalised to [0, 1]²) |
| `stress` | (N, 1) | von Mises stress at each node (MPa) |
| `corner` | (2,) | L-bracket corner position *(L-bracket only)* |
| `params` | (3,) | Hole centre + radius *(Plate-hole only)* |
| `sdf` | (N, 1) | Pre-computed SDF *(optional — computed on-the-fly if absent)* |

SDF is the minimum distance from each mesh node to the nearest boundary edge (positive inside the domain, zero on the boundary). Run `python compute_sdf.py --dataset both` to pre-compute and cache SDF values in the HDF5 files.

---

## Training Configuration

All three models are trained with **identical** hyperparameters for a fair comparison.

| Hyperparameter | Value |
|---|---|
| Preset | **M** |
| Epochs | 500 (early stopping) |
| Early stopping patience | 40 epochs |
| Learning rate | 3 × 10⁻⁴ |
| Optimizer | AdamW (weight decay 1 × 10⁻⁴) |
| LR scheduler | OneCycleLR — cosine, 10 % warm-up |
| Gradient clip | 0.5 |
| Training mode | `batched_all` — all nodes per geometry, padded to batch max |
| Mixed precision | AMP (CUDA) |
| Train / val split | 80 / 20, seed 42 |

### Preset M — architecture dimensions

Every component scales with the preset tier. All three models share the same M-tier dimensions:

| Component | Dimension |
|---|---|
| `latent_dim` | 160 |
| `pre_hidden` (PointNet++ pre-MLP) | [96, 96] |
| SA block 1 | out_ch=192, mlp_hidden=[96], max_k=40 |
| SA block 2 | out_ch=384, mlp_hidden=[192], max_k=40 |
| `gf_hidden` (global feature MLP) | [96] |
| `basis_dim` | 192 |
| `siren_hidden` (trunk SIREN hidden) | [384, 384] |
| `post_mlp_hidden` (post-fusion MLP) | [384, 384, 192] |

For `Point_DeepONet`, `branch_hidden` for the Vanilla PointNet MLP is [96, 192, 384].

Available preset tiers: **S** (latent 128) · **M** (latent 160) · **L** (latent 192) · **XL** (latent 256) · **XXL** (latent 384). All component widths scale proportionally with the tier.

---

## Usage

### Requirements

```bash
pip install torch numpy h5py scikit-learn
```

### Running on a SLURM cluster (recommended)

Each SLURM script submits one model to one GPU. `GPU0.py` automatically trains on both datasets sequentially and retries with smaller batch sizes on OOM errors.

```bash
sbatch run0.sh   # Point_DeepONet       → logs: GPU0.log
sbatch run1.sh   # Point++_wSDF         → logs: GPU1.log
sbatch run2.sh   # Point++_noSDF        → logs: GPU2.log
```

### Running directly

```bash
cd Point_DeepONet
python GPU0.py                # trains M preset on L-bracket then Plate-hole

# or call the training script directly for a single dataset:
python Training_script.py     # edit main() call at the bottom to select dataset/preset
```

### Selecting dataset or preset

Edit the `main()` call at the bottom of `Training_script.py`:

```python
if __name__ == "__main__":
    main("M", batch=8, dataset="L_bracket")    # L-bracket with M preset
    # main("M", batch=8, dataset="Plate_hole") # Plate-hole with M preset
```

Or via `GPU0.py`:

```python
PRESETS_GPU0 = [
    ["M", "L_bracket"],
    ["M", "Plate_hole"],
]
```

### Pre-computing SDF (optional)

The training scripts compute SDF on-the-fly if the `sdf` key is absent from the HDF5 file. To cache it for faster data loading:

```bash
python compute_sdf.py --dataset both
```

---

## Checkpoints

Best checkpoints are saved to `<model_folder>/Trained_models/` with the naming convention:

```
{geom_prefix}{model_name}_{arch_hash}.pt
```

Examples: `L-pnt_deeponet_m_a1b2c3d4.pt`, `H-pn2_wsdf_m_e5f6a7b8.pt`

Each checkpoint contains:

```python
{
    "model_state":      <state_dict>,
    "arch":             <arch_dict>,          # model config (if supported)
    "coord_center":     <tensor>,
    "coord_half_range": <tensor>,
    "sdf_mean":         <tensor or None>,     # wSDF / Point_DeepONet only
    "sdf_std":          <tensor or None>,
    "stress_mean":      <tensor>,
    "stress_std":       <tensor>,
    "best_val_loss":    <float>,
}
```

---

## Metrics logged during training

```
Epoch 001 | train MSE: 0.1234 | val MSE: 0.2345 | val MSE(MPa^2): 12.34 | R2(MPa): 0.8912 | lr: 3.00e-05 | epoch: 45.2s
```

- **train/val MSE** — normalised MSE (dimensionless)
- **val MSE (MPa²)** — MSE in physical stress units
- **R² (MPa)** — coefficient of determination in stress space

---

## Citation

```
[Add citation information here]
```

## License

[Add license information here]
