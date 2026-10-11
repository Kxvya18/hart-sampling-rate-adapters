# Quantization Plan for TSPulse HAR Model

## References
Based on peer-reviewed papers:
1. **Quantizing deep convolutional networks for efficient inference** (Krishnamoorthi, 2018, Google)
   - Per-channel weights + per-layer activations = SOTA quantization scheme
   - 8-bit INT8 achieves <2% accuracy drop (PTQ) to 1% (QAT)
   - 4x model size reduction, 2-3x CPU speedup, 10x on specialized HW

2. **Understanding and Overcoming the Challenges of Efficient Transformer Quantization** (Bondarenko et al., AAAI 2022)
   - Transformers have unique challenges: high dynamic activation ranges
   - Structured outliers in residual connections → activation quantization sensitive
   - Solutions: PTQ with clipping, per-embedding-group quantization, QAT

3. **Data-Free Quantization Through Weight Equalization and Bias Correction** (Nagel et al., ICCV 2019)
   - Weight equalization technique (no calibration data needed)
   - Bias correction to offset quantization error
   - SOTA for data-free 8-bit quantization

---

## Phase 1: POST-TRAINING QUANTIZATION (PTQ) - Quick Baseline
**Goal**: Establish 8-bit quantized baseline with minimal changes

### 1.1 Static PTQ (Per-Channel Weights + Per-Layer Activations)
- **Tool**: PyTorch's `torch.quantization` module (built-in)
- **Method**: Calibrate on validation set, then apply quantization
- **Expected results**: 
  - Model size: ~156KB → ~39KB (4x reduction)
  - Latency improvement: ~2-3x on CPU inference
  - Accuracy: <2% drop (baseline: 0.70 F1 → ~0.68 F1 expected)

**Steps**:
```python
# 1. Set model to eval mode
model.eval()

# 2. Insert quantization stubs
from torch.quantization import quantize_dynamic
# For activations: use QConfig with quantization-aware setup

# 3. Calibrate on 500-1000 samples from validation set
# (not training set to maintain LOSO separation)

# 4. Convert to quantized int8 model
quantized_model = torch.quantization.convert(model)

# 5. Evaluate on test sets (wisdm, motion_sense, har_sense)
# Measure: F1, latency, model size
```

**Why this first?**: 
- ✓ No retraining needed (10min total)
- ✓ Establishes upper bound on quantization impact
- ✓ Validates if full QAT is needed

---

## Phase 2: QUANTIZATION-AWARE TRAINING (QAT) - Higher Accuracy
**Goal**: Achieve 1% accuracy drop at 8-bit (if PTQ drops >2%)

### 2.1 Transformer-Specific QAT (per Bondarenko et al., 2022)
- **Problem**: Transformer attention activations have outliers → difficult to quantize
- **Solution**: Outlier clipping + per-embedding-group quantization

**Steps**:
```python
# 1. Load fine-tuned TSPulse model
model = TSPulseForHAR.from_pretrained(checkpoint_path)

# 2. Prepare for QAT with outlier-aware clipping
from torch.quantization import QConfig, default_qat_qconfig
model.qconfig = default_qat_qconfig  # or custom config
torch.quantization.prepare_qat(model, inplace=True)

# 3. Fine-tune for 5-10 epochs on LOSO training split
# (NOT validation/test - maintain LOSO integrity)
# Use lower LR (~1e-5) and shorter schedule
for epoch in range(5):
    train_one_epoch_quantized(model, train_loader)
    val_loss = evaluate(model, val_loader)
    if early_stopping_triggered:
        break

# 4. Convert to int8
torch.quantization.convert(model, inplace=True)

# 5. Evaluate on all 3 datasets
```

**Expected improvement over PTQ**:
- F1 recovery: 0.70 → 0.694 (1% drop instead of 2%)
- Training cost: ~30-60 min per dataset × 3 datasets = 2-3 hours total

---

## Phase 3: WEIGHT-ONLY QUANTIZATION (Optional, for Size-Critical Scenarios)
**Goal**: 8x model compression while keeping activations FP32 (if PTQ accuracy unacceptable)

### 3.1 INT8 Weight-Only Quantization
- **Benefit**: Weights 8x smaller, activations stay FP32 (fewer outlier issues)
- **Trade-off**: Less latency improvement than full INT8, but better accuracy
- **Tool**: PyTorch's `quantize_dynamic()` function

```python
model_wq = torch.quantization.quantize_dynamic(
    model, 
    qconfig_spec={torch.nn.Linear, torch.nn.LayerNorm},  # quantize these layers only
    dtype=torch.qint8
)
```

**Expected results**:
- Model size: ~156KB → ~78KB (2x reduction, not 4x)
- Accuracy: ~98-99% (minimal drop)
- Latency: ~1.5x improvement (memory bound instead of compute)

---

## Phase 4: HARDWARE-SPECIFIC OPTIMIZATION
**Goal**: Deploy on edge hardware (mobile/IoT)

### 4.1 ONNX Export + TensorRT/CoreML
```python
# Export quantized model for edge inference
torch.onnx.export(
    quantized_model,
    dummy_input,
    "model_int8.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13  # supports quantization operators
)

# For mobile: convert to CoreML (iOS) or TFLite (Android)
```

---

## Implementation Priority & Timeline

| Phase | Method | Effort | Accuracy | Speedup | Model Size | Priority |
|-------|--------|--------|----------|---------|------------|----------|
| 1 | Static PTQ | 10 min | -1.5% | 2-3x | 4x | **FIRST** |
| 2 | QAT (Transformer) | 2-3 hrs | -1% | 2-3x | 4x | **IF Phase 1 fails** |
| 3 | Weight-Only INT8 | 10 min | -0.5% | 1.5x | 2x | **Fallback** |
| 4 | ONNX+Deploy | 1 hr | - | - | - | **Deployment** |

---

## Evaluation Protocol (CRITICAL)

### Before Quantization
```python
# Establish baseline on all 3 datasets (LOSO)
baseline_results = {
    'wisdm': evaluate_on_loso_folds(...),      # F1=0.702
    'motion_sense': evaluate_on_loso_folds(...), # F1=0.577
    'har_sense': evaluate_on_loso_folds(...)     # F1=0.379
}
```

### After Quantization (PTQ)
```python
# Compare on SAME LOSO folds (no data leakage!)
ptq_results = {
    'wisdm': evaluate_on_same_loso_folds(...),
    'motion_sense': evaluate_on_same_loso_folds(...),
    'har_sense': evaluate_on_same_loso_folds(...)
}

# Calculate accuracy retention
for dataset in baseline_results:
    baseline_f1 = baseline_results[dataset]
    ptq_f1 = ptq_results[dataset]
    retention = (ptq_f1 / baseline_f1) * 100
    print(f"{dataset}: {retention:.1f}% retention")
```

### After QAT (if needed)
```python
# IMPORTANT: Calibrate on SEPARATE validation set
# (not training set used during fine-tuning)
# This avoids optimization overfitting to calibration distribution

qat_results = {
    'wisdm': evaluate_on_same_loso_folds(...),
    'motion_sense': evaluate_on_same_loso_folds(...),
    'har_sense': evaluate_on_same_loso_folds(...)
}
```

---

## Quantization Decision Tree

```
START: Evaluate PTQ
│
├─ Accuracy loss < 1.5% AND you're happy with 2-3x speedup?
│  └─ YES → USE PTQ (STOP HERE)
│
├─ NO → Try QAT (Transformer-specific outlier handling)
│  │
│  ├─ Accuracy loss now < 1% (after QAT fine-tuning)?
│  │  └─ YES → USE QAT
│  │  └─ NO → Fallback to Phase 3 (Weight-Only)
│
├─ Still not happy with accuracy?
│  └─ Weight-Only INT8 (2x size reduction, ~0.5% drop)
│
└─ Need hard deadline on model size?
   └─ 4-bit weights (aggressive, 8x compression, ~5-10% drop)
```

---

## Tools & Libraries

### Built-in (PyTorch)
- `torch.quantization` - Full PTQ/QAT support
- `torch.ao.quantization` - More recent API (PyTorch 1.13+)

### Third-Party (Recommended)
- **PyTorch Lightning** - Easier QAT integration
- **TensorFlow Lite** - Better edge deployment
- **NVIDIA TensorRT** - GPU inference optimization
- **Intel OpenVINO** - CPU optimization

### Validation
- **ONNX Runtime** - Cross-platform quantized inference
- **TensorFlow Lite** - Mobile profiling

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Outliers in attention weights | >5% F1 drop | Use transformer-specific clipping (Phase 2) |
| Activation range mismatch | Model collapse on OOD | Calibrate on diverse subjects (not single subject) |
| Calibration set distribution | Overfitting | Use validation set (separate from training) |
| Per-channel bias drift | Slow convergence in QAT | Use bias correction (Nagel et al., 2019) |
| Layer-specific precision needs | Uneven quantization | Try per-layer bitwidth tuning (advanced) |

---

## Success Criteria

✓ **Phase 1 (PTQ)**: F1 retention ≥98% (≤1% drop), 2-3x speedup  
✓ **Phase 2 (QAT)**: F1 retention ≥99% (≤1% drop), 2-3x speedup, <2hr training  
✓ **Phase 3 (Fallback)**: F1 retention ≥99.5%, 2x model size reduction  
✓ **Phase 4 (Deploy)**: Model runs on edge device with <10ms latency  

---

## Next Steps (Concrete)

1. **THIS WEEK**: Implement Phase 1 (Static PTQ)
   - Script: `quantize_ptq.py` 
   - Calibration: 1000 samples from validation folds
   - Evaluation: Report F1 on all 3 datasets

2. **IF NEEDED**: Implement Phase 2 (QAT)
   - Script: `quantize_qat.py`
   - Fine-tune: 5-10 epochs per dataset
   - Validation: Track F1 recovery

3. **Deployment**: Export to ONNX + TFLite

---

## References to Cite

```
@article{krishnamoorthi2018quantizing,
  title={Quantizing deep convolutional networks for efficient inference},
  author={Krishnamoorthi, Raghuraman},
  journal={arXiv preprint arXiv:1806.08342},
  year={2018}
}

@article{bondarenko2021understanding,
  title={Understanding and Overcoming the Challenges of Efficient Transformer Quantization},
  author={Bondarenko, Yelysei and Nagel, Markus and Blankevoort, Tijmen},
  journal={arXiv preprint arXiv:2109.12948},
  year={2021}
}

@article{nagel2019data,
  title={Data-free quantization through weight equalization and bias correction},
  author={Nagel, Markus and Van Baalen, Mart and Blankevoort, Tijmen and Welling, Max},
  booktitle={ICCV},
  year={2019}
}
```
