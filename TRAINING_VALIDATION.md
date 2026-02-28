# Input Pipeline Verification — wSDF and noSDF

This document traces the complete data flow for `Point++_DeepONet_wSDF` and `Point++_DeepONet_noSDF` to confirm that each model receives the correct inputs at every stage.

---

## noSDF — expected inputs: (x, y) only, no SDF anywhere

### Data loading
| Step | Code | Result shape | Notes |
|---|---|---|---|
| `load_h5_pointsets` | reads `points[N,2]` + `stress[N,1]`; stacks as `hstack([xy, stress])` | `[N, 3]` | no SDF loaded |
| `compute_global_normalization` | `t[:,:2]` = xy, `t[:,2:3]` = stress | — | ✅ stress is at column 2 |
| `GeomStressDataset.__getitem__` | `xy = t[:,:2]`; returns `{"points": xyn}` | `points: [N, 2]` | ✅ 2-channel |
| `AllNodesPadCollate` | stacks `pts=[N,2]` per sample | `geom_points: [B,N,2]`, `query_points: [B,N,2]` | ✅ |

### Model forward — `ScaledDiagramDeepONet`
| Step | Input | Notes |
|---|---|---|
| `encoder_cfg["sdf_ch"]` | not set → defaults to `0` | ✅ |
| `PointNet2Encoder2D.__init__` | `in_ch_pre = 2 + 0 = 2` → `pre = Linear(2, 96)` | ✅ |
| `PointNet2Encoder2D.forward(xyz=[B,N,2])` | `sdf_ch=0` → `coords=xyz`, no extra; `centers=coords=[B,N,2]` | ✅ 2-D spatial ops |
| `SetAbstraction` ball query | `centers=[B,N,2]` | ✅ 2-D distances |
| `SIREN` trunk | `trunk_in_ch = 2+0 = 2`; `q_flat=[B*Q,2]` → `SineLayer(2, 384)` | ✅ |

### Validation loop
```
pts = batch["points"]  →  [N, 2]
model(pts.unsqueeze(0), pts.unsqueeze(0))  →  model([1,N,2], [1,N,2])  ✅
```

---

## wSDF — expected inputs: (x, y) for branch spatial ops; (x, y, sdf) for branch features and trunk SIREN

### Data loading
| Step | Code | Result shape | Notes |
|---|---|---|---|
| `load_h5_pointsets` | reads `points[N,2]`, `sdf[N,1]`, `stress[N,1]`; stacks as `hstack([xy, sdf, stress])` | `[N, 4]` | SDF in column 2 |
| `compute_global_normalization` | `t[:,:2]`=xy, `t[:,2:3]`=sdf, `t[:,3:4]`=stress | — | ✅ layout matches |
| `GeomStressDataset.__getitem__` | `pts_with_sdf = cat([xyn, sdfn])`; returns `{"points": pts_with_sdf}` | `points: [N, 3]` | ✅ 3-channel with SDF |
| `AllNodesPadCollate` | stacks `pts=[N,3]` per sample | `geom_points: [B,N,3]`, `query_points: [B,N,3]` | ✅ |

### Model forward — `ScaledDiagramDeepONet`
| Step | Input | Notes |
|---|---|---|
| `encoder_cfg["sdf_ch"]` | set to `1` in Training_script | ✅ |
| `PointNet2Encoder2D.__init__` | `in_ch_pre = 2 + 1 = 3` → `pre = Linear(3, 96)` | ✅ branch pre-MLP sees SDF feature |
| `PointNet2Encoder2D.forward(xyz=[B,N,3])` | `sdf_ch=1` → `coords=xyz[...,:2]=[B,N,2]`; `extra=xyz[...,2:3]=[B,N,1]`; `x_in=cat([coords,extra])=[B,N,3]` → `pre(x_in)` | ✅ |
| `centers = coords` | `[B,N,2]` — always 2-D for ball query / FPS | ✅ spatial ops are 2-D only |
| `SetAbstraction` ball query | `centers=[B,N,2]` | ✅ |
| `SIREN` trunk | `trunk_in_ch = 2+1 = 3`; `q_flat=[B*Q,3]` → `SineLayer(3, 384)` | ✅ trunk sees SDF |

### Validation loop
```
pts = batch["points"]  →  [N, 3]  (x_norm, y_norm, sdf_norm)
model(pts.unsqueeze(0), pts.unsqueeze(0))  →  model([1,N,3], [1,N,3])  ✅
```

---

## Summary

| | noSDF branch encoder | noSDF trunk SIREN | wSDF branch encoder | wSDF trunk SIREN |
|---|---|---|---|---|
| Spatial ops (FPS, ball query) | (x, y) | — | (x, y) | — |
| Per-point MLP features | (x, y) | — | (x, y, sdf) | — |
| SIREN input channels | — | (x, y) | — | (x, y, sdf) |

✅ `noSDF` uses (x, y) only throughout.  
✅ `wSDF` uses SDF as a per-point feature in the branch pre-MLP **and** as an additional query coordinate in the trunk SIREN. The PointNet++ spatial operations (FPS, ball query, relative positions) always use 2-D (x, y) regardless of SDF.

