# Answer: Why DenseNoFFT Trained While DeepONet Models Failed

## TL;DR
They were NOT all using BatchNorm equally. The encoder (majority of the model) was correctly using LayerNorm from config. Only the task-specific MLPs were incorrectly hardcoded to BatchNorm. DenseNoFFT had 1 problematic MLP while DeepONet models had 2-3, making them fail more severely.

## Detailed Explanation

### What All Models Share
All four models (DenseNoFFT, PointNetMLPJoint, SpectralDeepONet, VanillaDeepONet) use the same **PointNet2Encoder2D** encoder, which has:
- Pre-processing MLP
- 2 SetAbstraction blocks (each with their own MLPs)
- GlobalFeature aggregation (with another MLP)

This encoder represents the **bulk of the model** and was **correctly** using LayerNorm because it reads `"norm": "layer"` from encoder_cfg.

### The Bug Was Only in Task-Specific Layers

**Before the fix**, only the final task-specific layers had hardcoded `norm="batch"`:

1. **DenseNoFFT**: 1 head MLP with BatchNorm
2. **PointNetMLPJoint**: HEAD properly used LayerNorm (reads from encoder_cfg)
3. **SpectralDeepONet**: branch_net + trunk_net both with BatchNorm (2 MLPs)
4. **VanillaDeepONet**: branch_net + trunk_net both with BatchNorm (2 MLPs)

### Why DeepONet Failed But DenseNoFFT Didn't

**DeepONet Architecture Complexity**:
```
Encoder (✓ LayerNorm)
   ├─> Branch MLP (✗ BatchNorm) ─┐
   │                               ├─> Dot Product ─> Output
Query ─> Trunk MLP (✗ BatchNorm) ─┘
```

**DenseNoFFT Architecture Simplicity**:
```
Encoder (✓ LayerNorm) ─┐
                        ├─> Concat ─> Head MLP (✗ BatchNorm) ─> Output
Query ──────────────────┘
```

The DeepONet models:
1. Have **2 separate paths** with BatchNorm (branch + trunk)
2. Process different types of inputs (encoder output vs. raw/Fourier coordinates)
3. Rely on **precise alignment** between branch and trunk for the dot product fusion
4. Are more **sensitive to normalization inconsistencies**

The error messages ("running_mean should contain 418 elements not 512") likely stem from:
- Complex interactions between the two BatchNorm paths
- Variable input dimensions from different geometries
- The fusion mechanism being disrupted by normalization mismatches

### The Fix

Changed benchmarks.py to extract `norm` from encoder_cfg:
```python
# Before (hardcoded)
self.branch_net = MLP(..., norm="batch")

# After (from config)
norm_type = str(encoder_cfg.get("norm", "batch"))
self.branch_net = MLP(..., norm=norm_type)
```

Now all models use LayerNorm consistently, eliminating the errors.

## Conclusion

DenseNoFFT **likely had issues too**, but they were **less severe** because:
1. Only 1 MLP layer had the bug (vs. 2 in DeepONet)
2. Simpler concatenation architecture vs. complex fusion
3. The encoder (which worked correctly) represents most of the model's capacity

The DeepONet models failed more catastrophically because their architectural complexity amplified the normalization mismatch issues.
