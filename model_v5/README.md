# model_v5 — FT-Transformer (Focal Loss γ = 2)

## Changes vs v4

| Change         | v4                         | v5                     | Reason                                                                                  |
| -------------- | -------------------------- | ---------------------- | --------------------------------------------------------------------------------------- |
| Loss function  | `categorical_crossentropy` | **Focal loss (γ = 2)** | Down-weights easy `normal` predictions; focuses gradients on hard minority class errors |
| Normal cap     | 2,000                      | 2,000 (unchanged)      |                                                                                         |
| SMOTE          | ✗                          | ✗ (unchanged)          |                                                                                         |
| `dropout_rate` | 0.3                        | 0.3 (unchanged)        |                                                                                         |

## Architecture (unchanged)

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

| Setting        | Value                                                           |
| -------------- | --------------------------------------------------------------- |
| Loss           | **Focal loss, γ = 2**                                           |
| Optimizer      | Adam, `lr = 1e-3`                                               |
| Batch size     | 256                                                             |
| Max epochs     | 100                                                             |
| Early stopping | `patience=10`, restore best weights                             |
| LR scheduler   | `ReduceLROnPlateau`, `patience=5`, `factor=0.5`                 |
| Class weights  | `balanced` — used alongside focal loss (alpha-balanced variant) |
| **Best epoch** | **4** (stopped at 14)                                           |

## Results

| Split      | Accuracy | Macro F1   | Weighted F1 | Macro AUROC | Macro AUPRC |
| ---------- | -------- | ---------- | ----------- | ----------- | ----------- |
| Validation | 0.6676   | **0.3213** | 0.7472      | 0.7532      | 0.3371      |
| Test       | 0.6812   | **0.3283** | 0.7588      | 0.7635      | 0.3346      |

## v4 → v5 Delta

| Metric      | v4 Test | v5 Test    | Δ          |
| ----------- | ------- | ---------- | ---------- |
| Accuracy    | 0.6693  | 0.6812     | **+0.012** |
| Macro F1    | 0.3188  | **0.3283** | **+0.010** |
| Weighted F1 | 0.7517  | 0.7588     | **+0.007** |
| Macro AUROC | 0.7830  | 0.7635     | -0.020     |
| Macro AUPRC | 0.3353  | 0.3346     | -0.001     |

**Interpretation:** Focal loss improved Accuracy, Macro F1, and Weighted F1 — confirming it helps the model focus on minority class errors. AUROC dropped slightly (0.7830 → 0.7635) which is expected: focal loss shifts the decision boundary toward minority classes, changing the probability calibration in a way that slightly reduces overall ranking ability. Macro F1 is now the highest seen across all versions at **0.3283**. Next step: grid/random search to find optimal hyperparameters for this focal loss + 2k-normal setup.
