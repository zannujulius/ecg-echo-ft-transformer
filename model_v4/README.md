# model_v4 — FT-Transformer (Normal cap 2,000 — No SMOTE)

## Changes vs v3

| Change           | v3            | v4                  | Reason                                                        |
| ---------------- | ------------- | ------------------- | ------------------------------------------------------------- |
| Normal cap       | 3,000 + SMOTE | **2,000, no SMOTE** | 100% real data; near-balanced ratio without synthetic samples |
| `dropout_rate`   | 0.3           | 0.3 (unchanged)     | Carried from v2                                               |
| Training shuffle | ✓             | ✓ (unchanged)       | Carried from v2                                               |

## Training Data

All samples are **real patients** — no synthetic augmentation.

| Class     | Count     | %     |
| --------- | --------- | ----- |
| Normal    | 2,000     | 29.0% |
| Mild      | 1,974     | 28.6% |
| Severe    | 1,580     | 22.9% |
| Moderate  | 1,339     | 19.4% |
| **Total** | **6,893** |       |

## Architecture (unchanged from v2/v3)

| Hyperparameter           | Value                    |
| ------------------------ | ------------------------ |
| `embedding_dim`          | 32                       |
| `num_transformer_blocks` | 2                        |
| `num_heads`              | 4                        |
| `dense_dim`              | 32                       |
| `dropout_rate`           | 0.3                      |
| Pooling                  | `GlobalAveragePooling1D` |
| Total parameters         | 13,828                   |

## Training

| Setting        | Value                                           |
| -------------- | ----------------------------------------------- |
| Loss           | `categorical_crossentropy`                      |
| Optimizer      | Adam, `lr = 1e-3`                               |
| Batch size     | 256                                             |
| Max epochs     | 100                                             |
| Early stopping | `patience=10`, restore best weights             |
| LR scheduler   | `ReduceLROnPlateau`, `patience=5`, `factor=0.5` |
| Class weights  | `balanced` from 2k-normal train distribution    |
| **Best epoch** | **4** (stopped at 14)                           |

## Results

| Split      | Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC |
| ---------- | -------- | -------- | ----------- | ----------- | ----------- |
| Validation | 0.6579   | 0.3107   | 0.7408      | 0.7686      | 0.3349      |
| Test       | 0.6693   | 0.3188   | 0.7517      | **0.7830**  | 0.3353      |

## v3 → v4 Delta

| Metric      | v3 Test | v4 Test    | Δ               |
| ----------- | ------- | ---------- | --------------- |
| Accuracy    | 0.6758  | 0.6693     | -0.007          |
| Macro F1    | 0.3186  | **0.3188** | +0.000 (≈ same) |
| Weighted F1 | 0.7553  | 0.7517     | -0.004          |
| Macro AUROC | 0.7757  | **0.7830** | **+0.007**      |
| Macro AUPRC | 0.3248  | 0.3353     | **+0.010**      |

**Interpretation:** v4 matches v3's Macro F1 (essentially identical at 0.3188 vs 0.3186) while using 100% real data and a smaller training set (6,893 vs 12,000). AUROC and AUPRC are slightly better than v3. This confirms the SMOTE synthetic samples in v3 weren't adding meaningful signal — real data quality wins over synthetic quantity. The model is now the best baseline for adding focal loss in v5.
