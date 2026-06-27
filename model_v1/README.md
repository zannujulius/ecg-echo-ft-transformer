# AS Severity Classification — FT-Transformer Experiments

**Dataset:** EchoNext (PhysioNet v1.1.1) — https://physionet.org/content/echonext/1.1.1/  
**Task:** 4-class classification of Aortic Stenosis severity: `normal` · `mild` · `moderate` · `severe`  
**Notebook:** `model.ipynb` (requires `venv` — run `pip install -r requirements.txt`)

---

## Feature Groups

| Group        | Features                                                                                   | Count  |
| ------------ | ------------------------------------------------------------------------------------------ | ------ |
| ECG          | `pr_interval`, `qrs_duration`, `qt_corrected`, `ventricular_rate`, `atrial_rate`           | 5      |
| Echo         | `ivs_measurement`, `lvpw_measurement`, `pasp_value`, `tr_max_velocity_value`, `lvef_value` | 5      |
| Demographics | `age_at_ecg`, `sex`                                                                        | 2      |
| **Total**    |                                                                                            | **12** |

---

## Data Splits

| Split               | Normal | Mild  | Moderate | Severe | Total  |
| ------------------- | ------ | ----- | -------- | ------ | ------ |
| Train (balanced)    | 3,000  | 1,974 | 1,339    | 1,580  | 7,893  |
| Val (untouched)     | 3,351  | 136   | 101      | 151    | 3,739  |
| Test (untouched)    | 3,999  | 151   | 114      | 172    | 4,436  |
| No-split (excluded) | 13,914 | 603   | 303      | 294    | 15,114 |

- `normal` class capped at **3,000** via random undersampling (`random_state=42`) on the train split only
- Val and test kept at natural (imbalanced) distribution to reflect real-world AS prevalence
- `no_split` rows excluded — split origin unknown, leakage risk

## Preprocessing

| Step               | Detail                                                                    |
| ------------------ | ------------------------------------------------------------------------- |
| Missing imputation | `SimpleImputer(strategy='median')` on 11 numeric features                 |
| Scaling            | `StandardScaler()`                                                        |
| Pipeline fit       | Train only — transformed onto val/test                                    |
| Categorical        | `sex` passed as raw string; handled by `StringLookup` inside the TF graph |

---

## Model Versions

### model_v1

**Architecture — FT-Transformer**

| Hyperparameter           | Value                    |
| ------------------------ | ------------------------ |
| `embedding_dim`          | 32                       |
| `num_transformer_blocks` | 2                        |
| `num_heads`              | 4                        |
| `dense_dim` (FFN inner)  | 32                       |
| `dropout_rate`           | 0.1                      |
| Pooling                  | `GlobalAveragePooling1D` |
| Output activation        | `softmax`                |
| Total parameters         | 13,828                   |

**Training**

| Setting        | Value                                                          |
| -------------- | -------------------------------------------------------------- |
| Loss           | `categorical_crossentropy`                                     |
| Optimizer      | Adam, `lr = 1e-3`                                              |
| Batch size     | 256                                                            |
| Max epochs     | 100                                                            |
| Early stopping | `patience=10`, monitor `val_loss`, restore best weights        |
| LR scheduler   | `ReduceLROnPlateau`, `patience=5`, `factor=0.5`, `min_lr=1e-6` |
| Class weights  | `balanced` (computed from post-undersampling train)            |

**Class weights used**

| Class    | Train count | Weight |
| -------- | ----------- | ------ |
| Normal   | 3,000       | 0.6577 |
| Mild     | 1,974       | 0.9996 |
| Severe   | 1,580       | 1.2489 |
| Moderate | 1,339       | 1.4737 |

**Results**

| Split      | Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC |
| ---------- | -------- | -------- | ----------- | ----------- | ----------- |
| Validation | 0.5764   | 0.2986   | 0.6811      | 0.7362      | 0.3255      |
| Test       | 0.5780   | 0.2837   | 0.6867      | 0.7364      | 0.3115      |

**Interpretation:** Macro F1 (~0.29) is poor — the model struggles with the minority classes (mild/moderate/severe) on the imbalanced val/test sets. Macro AUROC (~0.74) suggests reasonable discriminative ability but the decision boundary is poorly calibrated. The gap between weighted F1 (~0.68) and macro F1 (~0.29) confirms the model is biased toward the majority `normal` class. Ablation directions below.

**Output files** (`model_v1/`)

| File                                | Description                                       |
| ----------------------------------- | ------------------------------------------------- |
| `class_distribution.png`            | Raw AS class counts in full filtered dataset      |
| `class_distribution_comparison.png` | Before vs after naive undersampling (exploratory) |
| `split_raw_distribution.png`        | Per-split class counts before rebalancing         |
| `split_final_distribution.png`      | Train (balanced) / Val / Test side-by-side        |
| `missing_values.png`                | Missing data % per feature                        |
| `ecg_distributions.png`             | ECG feature KDE by severity class                 |
| `echo_distributions.png`            | Echo feature KDE by severity class                |
| `demo_distributions.png`            | Age boxplot + sex stacked bar by severity         |
| `training_history.png`              | Train/val loss and accuracy curves                |
| `confusion_matrix.png`              | 4×4 confusion matrix (counts + row-normalised %)  |
| `roc_pr_curves.png`                 | Per-class ROC and Precision-Recall curves         |

---

## Ablation Ideas (for future versions)

| Area           | Change to try                                               | Hypothesis                                              |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------- |
| Architecture   | Increase `embedding_dim` to 64 or 128                       | Richer feature representations                          |
| Architecture   | Increase `num_transformer_blocks` to 3–4                    | Deeper cross-feature attention                          |
| Architecture   | Increase `num_heads` to 8                                   | More attention patterns                                 |
| Regularisation | Increase `dropout_rate` to 0.2–0.3                          | Reduce overfitting to train                             |
| Data           | Add shuffling to `DataLoader`                               | Break batch ordering bias                               |
| Data           | SMOTE or class-conditional augmentation on minority classes | Better minority class coverage than undersampling alone |
| Training       | Focal loss instead of crossentropy                          | Penalises easy majority-class predictions more          |
| Training       | Label smoothing                                             | Prevent overconfident softmax predictions               |
| Features       | Add feature interaction terms                               | Explicit clinical priors (e.g. TR velocity × LVEF)      |
