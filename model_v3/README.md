# model_v3 — FT-Transformer (SMOTE on Minority Classes)

## Changes vs v2

| Change | v2 | v3 | Reason |
|---|---|---|---|
| SMOTE on minority classes | ✗ | ✓ SMOTENC (`k_neighbors=5`) | Mild/moderate/severe under-represented even after undersampling |
| `dropout_rate` | 0.3 | 0.3 (unchanged) | Carried from v2 |
| Training shuffle | ✓ | ✓ (unchanged) | Carried from v2 |

## Training Data After SMOTE

| Class | Before SMOTE | After SMOTE |
|---|---|---|
| Normal | 3,000 | 3,000 (unchanged) |
| Mild | 1,974 | **3,000** |
| Moderate | 1,339 | **3,000** |
| Severe | 1,580 | **3,000** |
| **Total** | **7,893** | **12,000** |

## Architecture (unchanged from v2)

| Hyperparameter | Value |
|---|---|
| `embedding_dim` | 32 |
| `num_transformer_blocks` | 2 |
| `num_heads` | 4 |
| `dense_dim` | 32 |
| `dropout_rate` | 0.3 |
| Pooling | `GlobalAveragePooling1D` |
| Total parameters | 13,828 |

## Training

| Setting | Value |
|---|---|
| Loss | `categorical_crossentropy` |
| Optimizer | Adam, `lr = 1e-3` |
| Batch size | 256 |
| Max epochs | 100 |
| Early stopping | `patience=10`, restore best weights |
| LR scheduler | `ReduceLROnPlateau`, `patience=5`, `factor=0.5` |
| Class weights | All ≈ 1.0 (SMOTE balanced all classes equally) |
| **Best epoch** | **4** (stopped at 14) |

## Results

| Split | Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC |
|---|---|---|---|---|---|
| Validation | 0.6654 | 0.3121 | 0.7447 | 0.7674 | 0.3306 |
| Test | 0.6758 | 0.3186 | 0.7553 | 0.7757 | 0.3248 |

## v2 → v3 Delta

| Metric | v2 Test | v3 Test | Δ |
|---|---|---|---|
| Accuracy | 0.6844 | 0.6758 | -0.009 |
| Macro F1 | 0.3044 | **0.3186** | **+0.014** |
| Weighted F1 | 0.7590 | 0.7553 | -0.004 |
| Macro AUROC | 0.7774 | 0.7757 | -0.002 |
| Macro AUPRC | 0.3327 | 0.3248 | -0.008 |

**Interpretation:** SMOTE improved Macro F1 (+0.014) — the metric most sensitive to minority class performance — while accuracy and AUROC dropped very slightly. This is the expected trade-off: the model is now predicting more minority class cases (better minority recall) at the cost of slightly more false positives on the dominant `normal` class. Best epoch was 4, meaning SMOTE made the model converge faster but also plateau earlier.
