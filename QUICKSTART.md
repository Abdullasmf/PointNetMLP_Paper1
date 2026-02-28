# Quick-Start Reference

## Models at a glance

| Folder | Encoder | Trunk SDF? | Checkpoint prefix |
|---|---|---|---|
| `Point_DeepONet` | Vanilla PointNet | ✅ (x, y, sdf) | `pnt_deeponet_` |
| `Point++_DeepONet_noSDF` | PointNet++ | ❌ (x, y only) | `pn2_nosdf_` |
| `Point++_DeepONet_wSDF` | PointNet++ | ✅ (x, y, sdf) | `pn2_wsdf_` |

All three are trained at the **M preset** with identical hyperparameters.

---

## Standardised training settings

```python
preset              = "M"
epochs              = 500          # with early stopping
early_stopping      = 40 epochs
lr                  = 3e-4
weight_decay        = 1e-4
optimizer           = AdamW
scheduler           = OneCycleLR (cosine, 10 % warm-up)
grad_clip           = 0.5
train_mode          = "batched_all"
amp                 = True (CUDA)
train_val_split     = 0.8 / 0.2
random_seed         = 42
```

---

## Running on a SLURM cluster

```bash
sbatch run0.sh   # Point_DeepONet       → GPU0.log
sbatch run1.sh   # Point++_wSDF         → GPU1.log
sbatch run2.sh   # Point++_noSDF        → GPU2.log
```

Each script activates the `MLEnv` conda environment and calls `GPU0.py` inside the relevant model folder. `GPU0.py` runs preset **M** on both datasets (`L_bracket` then `Plate_hole`) with automatic batch-size fallback on OOM.

---

## Running directly (no SLURM)

```bash
# From the repo root
cd Point_DeepONet          && python GPU0.py
cd ../Point++_DeepONet_wSDF  && python GPU0.py
cd ../Point++_DeepONet_noSDF && python GPU0.py
```

To target a single dataset or a different preset, edit the `main()` call at the bottom of `Training_script.py`:

```python
if __name__ == "__main__":
    main("M", batch=8, dataset="L_bracket")
```

---

## SDF pre-computation (optional)

SDF is computed analytically on-the-fly if not stored. To cache it:

```bash
python compute_sdf.py --dataset both
```

---

## Checkpoint locations

```
Point_DeepONet/Trained_models/L-pnt_deeponet_m_<hash>.pt
Point_DeepONet/Trained_models/H-pnt_deeponet_m_<hash>.pt

Point++_DeepONet_wSDF/Trained_models/L-pn2_wsdf_m_<hash>.pt
Point++_DeepONet_wSDF/Trained_models/H-pn2_wsdf_m_<hash>.pt

Point++_DeepONet_noSDF/Trained_models/L-pn2_nosdf_m_<hash>.pt
Point++_DeepONet_noSDF/Trained_models/H-pn2_nosdf_m_<hash>.pt
```

Each checkpoint stores the model state dict, full normalisation stats, and best validation loss.

---

## Comparing results

After training, compare across models:

| Metric | Lower/Higher is better |
|---|---|
| val MSE (MPa²) | lower |
| R² (MPa space) | higher (max 1.0) |

Training logs print per-epoch:
```
Epoch 001 | train MSE: 0.1234 | val MSE: 0.2345 | val MSE(MPa^2): 12.34 | R2(MPa): 0.8912 | lr: 3.00e-05 | epoch: 45.2s
```
