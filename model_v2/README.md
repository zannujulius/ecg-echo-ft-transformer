# model_v2 — FT-Transformer (Shuffle Fix + Dropout 0.3)

## Changes vs v1

| Change           | v1                 | v2                                                     | Reason                                                                    |
| ---------------- | ------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| Training shuffle | ✗ off (silent bug) | ✓ `buffer_size=7,893`, `reshuffle_each_iteration=True` | Batches were always in identical order every epoch — broke generalisation |
| `dropout_rate`   | 0.1                | **0.3**                                                | v1 training curve diverged from ~epoch 40, clear overfitting              |
| `embedding_dim`  | 32                 | 32 (unchanged)                                         | More capacity would worsen overfitting, not fix it                        |

---

## Architecture

| Hyperparameter           | Value                    |
| ------------------------ | ------------------------ |
| `embedding_dim`          | 32                       |
| `num_transformer_blocks` | 2                        |
| `num_heads`              | 4                        |
| `dense_dim` (FFN inner)  | 32                       |
| `dropout_rate`           | **0.3**                  |
| Pooling                  | `GlobalAveragePooling1D` |
| Output activation        | `softmax`                |
| Total parameters         | 13,828                   |

## Training

| Setting        | Value                                                          |
| -------------- | -------------------------------------------------------------- |
| Loss           | `categorical_crossentropy`                                     |
| Optimizer      | Adam, `lr = 1e-3`                                              |
| Batch size     | 256                                                            |
| Max epochs     | 100                                                            |
| Early stopping | `patience=10`, monitor `val_loss`, restore best weights        |
| LR scheduler   | `ReduceLROnPlateau`, `patience=5`, `factor=0.5`, `min_lr=1e-6` |
| Class weights  | `balanced` (same as v1)                                        |
| **Best epoch** | **7** (early stopping triggered at epoch 17)                   |

> v1 ran all 100 epochs without early stopping. v2 converged and stopped at epoch 17 — a much healthier training curve.

---

## Results

| Split      | Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC |
| ---------- | -------- | -------- | ----------- | ----------- | ----------- |
| Validation | 0.6737   | 0.2994   | 0.7483      | 0.7611      | 0.3371      |
| Test       | 0.6844   | 0.3044   | 0.7590      | 0.7774      | 0.3327      |

## v1 → v2 Delta

| Metric      | v1 Test | v2 Test | Δ          |
| ----------- | ------- | ------- | ---------- |
| Accuracy    | 0.5780  | 0.6844  | **+0.106** |
| Macro F1    | 0.2837  | 0.3044  | **+0.021** |
| Weighted F1 | 0.6867  | 0.7590  | **+0.072** |
| Macro AUROC | 0.7364  | 0.7774  | **+0.041** |
| Macro AUPRC | 0.3115  | 0.3327  | **+0.021** |

All metrics improved. The shuffle fix had the largest impact — accuracy gained +0.10 in a single change.  
Macro F1 (~0.30) remains low, indicating minority classes (mild/moderate/severe) are still hard to separate on the imbalanced test set.

---

## Output Files

| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `training_history.png` | Train/val loss and accuracy — early stopping visible    |
| `confusion_matrix.png` | 4×4 confusion matrix (counts + row-normalised %)        |
| `roc_pr_curves.png`    | Per-class ROC and PR curves with macro AUROC/AUPRC      |
| `split_*.png`          | Split distribution charts (same as v1 — data unchanged) |

---

## Next Ablation Ideas for v3

| Idea                                     | Hypothesis                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| Focal loss (γ=2) instead of crossentropy | Penalises confident wrong predictions harder — may boost minority class recall |
| Increase `num_transformer_blocks` to 3   | Deeper cross-feature attention with regularisation now in place                |
| SMOTE on minority classes in train       | Better minority coverage than undersampling alone                              |
| Lower LR to `5e-4`                       | Model converged very fast (epoch 7) — smaller LR may find better optimum       |
