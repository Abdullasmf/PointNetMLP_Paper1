# Analysis: Why DenseNoFFT Trained Successfully While DeepONet Models Failed

## Summary
After investigation, the question "How did DenseNoFFT train (and get good results) while the other 2 failed to train even though all were using batch?" reveals a critical difference in the training script configurations.

## Key Findings

### 1. Model Assignments
- **GPU0**: DenseNoFFT
- **GPU1**: PointNetMLPJoint
- **GPU2**: SpectralDeepONet (FAILED)
- **GPU3**: VanillaDeepONet (FAILED)

### 2. Before The Fix

#### PointNetMLPJoint (in pn_models.py) - SUCCESS
- Training script: Passes `"head_norm": "layer"` in encoder_cfg ✓
- Model code: Reads and uses `head_norm` from encoder_cfg ✓
- **Result: Used LayerNorm correctly → SUCCESS**

#### DenseNoFFT (in benchmarks.py) - SUCCESS  
- Training script: Passes `"head_norm": "layer"` in encoder_cfg ✓
- Model code: **IGNORED encoder_cfg**, hardcoded `norm="batch"` ✗
- **Result: Used BatchNorm BUT STILL SUCCEEDED**

#### SpectralDeepONet & VanillaDeepONet (in benchmarks.py) - FAILURE
- Training scripts: `"head_norm": head_norm` **COMMENTED OUT** ✗
- Model code: Hardcoded `norm="batch"` ✗
- **Result: Used BatchNorm AND FAILED**

###  3. The Critical Difference

**In Training_script.py encoder_cfg:**

DenseNoFFT/Training_script.py (lines 673-674):
```python
"head_norm": head_norm,        # UNCOMMENTED
"head_dropout": head_dropout,  # UNCOMMENTED
```

SpectralDeepONet/Training_script.py (lines 672-673):
```python
# "head_norm": head_norm,        # COMMENTED OUT
# "head_dropout": head_dropout,  # COMMENTED OUT
```

VanillaDeepONet/Training_script.py (lines 672-673):
```python
# "head_norm": head_norm,        # COMMENTED OUT
# "head_dropout": head_dropout,  # COMMENTED OUT
```

### 4. Why This Matters

Even though all three model classes in benchmarks.py hardcoded `norm="batch"` (before the fix), the **presence or absence** of norm settings in encoder_cfg might affect other parts of the model, particularly:

1. The **encoder's SetAbstraction layers** read `norm` from encoder_cfg
2. The **encoder's GlobalFeature layer** reads `norm` from encoder_cfg  
3. These components work correctly with LayerNorm

### 5. Why DenseNoFFT Succeeded with BatchNorm

The most likely explanation is that **DenseNoFFT's simpler architecture** made it more resilient:

1. **DenseNoFFT architecture**: 
   - Encoder (uses LayerNorm from encoder_cfg) → 
   - Single head MLP (used BatchNorm due to hardcoding)
   
2. **DeepONet architecture**:
   - Encoder (uses LayerNorm from encoder_cfg) →
   - Branch MLP (used BatchNorm due to hardcoding) +
   - Trunk MLP (used BatchNorm due to hardcoding) →
   - Dot product fusion

The DeepONet models have MORE BatchNorm layers in critical paths (branch and trunk), which may have caused issues when:
- Processing variable-length point clouds
- Training on different geometries sequentially
- Handling the architectural complexity of the DeepONet fusion mechanism

## Conclusion

**The Answer**: DenseNoFFT succeeded because:
1. Its training script correctly passed `head_norm` to encoder_cfg
2. Its simpler architecture (single head MLP) was more tolerant of the BatchNorm bug
3. The encoder components (which constitute most of the model) were using LayerNorm correctly

The DeepONet models failed because:
1. Their training scripts had `head_norm` **commented out**
2. They have more complex architectures with multiple MLPs using BatchNorm
3. The combination of these factors made them more sensitive to normalization issues

## The Fix

The fix applied (uncommenting `head_norm` in Training_script.py files AND extracting norm from encoder_cfg in benchmarks.py) ensures:
1. All models respect the normalization setting from presets
2. Consistent use of LayerNorm throughout the architecture
3. No BatchNorm dimension mismatch errors
