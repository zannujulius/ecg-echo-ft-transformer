# model_v6 — FT-Transformer (LR 1e-4 + Dropout 0.2)

## Changes vs v5

| Change | v5 | v6 | Reason |
|---|---|---|---|
| Learning rate | 1e-3 | **1e-4** | v5 converged at epoch 4 — hoped lower LR would allow deeper exploration |
| `dropout_rate` | 0.3 | **0.2** | Focal loss already regularises; reduce double-suppression of training signal |
| Loss | Focal loss (γ=2) | Focal loss (γ=2, unchanged) | |
| Normal cap | 2,000 | 2,000 (unchanged) | |

## Training

| Setting | Value |
|---|---|
| Loss | Focal loss, γ = 2 |
| Optimizer | Adam, `lr = 1e-4` |
| Batch size | 256 |
| Max epochs | 100 |
| Early stopping | `patience=10`, restore best weights |
| LR scheduler | `ReduceLROnPlateau`, `patience=5`, `factor=0.5`, `min_lr=1e-7` |
| Class weights | `balanced` |
| **Best epoch** | **14** (stopped at 24) |

## Results

| Split | Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC |
|---|---|---|---|---|---|
| Validation | 0.6194 | 0.3085 | 0.7139 | 0.7414 | 0.3275 |
| Test | 0.6202 | 0.3132 | 0.7177 | 0.7594 | 0.3307 |

## v5 → v6 Delta

| Metric | v5 Test | v6 Test | Δ |
|---|---|---|---|
| Accuracy | 0.6812 | 0.6202 | **-0.061** |
| Macro F1 | 0.3283 | 0.3132 | **-0.015** |
| Weighted F1 | 0.7588 | 0.7177 | **-0.041** |
| Macro AUROC | 0.7635 | 0.7594 | -0.004 |
| Macro AUPRC | 0.3346 | 0.3307 | -0.004 |

## Interpretation

The hypothesis was partially correct — best epoch shifted from 4 to **14**, confirming the lower LR did allow training to run longer before early stopping. However, all metrics degraded.

**Why it didn't help:**
- With only 27 training batches per epoch and 6,893 samples, the gradient signal is already well-aggregated at each step. `lr=1e-3` was efficient precisely because each update carries meaningful signal. Reducing to `1e-4` slowed convergence without finding a better region.
- The train accuracy at epoch 14 (0.466) is lower than v5's train accuracy at epoch 4 (~0.496), meaning the model didn't actually learn better representations — it just took longer to reach the same (or worse) point.
- The train/val accuracy inversion persists, showing dropout=0.2 alone didn't solve the regularisation issue with focal loss.

**Conclusion:** `lr=1e-3` is the right scale for this dataset and batch size. **v5 remains the best configuration.** The next logical step is hyperparameter search (embedding_dim, num_heads, num_transformer_blocks) on the v5 setup.
