# AS Severity Classification — FT-Transformer Ablation Tracker

**Dataset:** EchoNext (PhysioNet v1.1.1) — https://physionet.org/content/echonext/1.1.1/  
**Task:** 4-class classification of Aortic Stenosis severity: `normal` · `mild` · `moderate` · `severe`  
**Notebook per version:** `model_vN/model.ipynb` — run with `venv` (`pip install -r requirements.txt`)

---

## Feature Groups (all versions)

| Group | Features | Type |
|---|---|---|
| ECG | `pr_interval`, `qrs_duration`, `qt_corrected`, `ventricular_rate`, `atrial_rate` | Numeric |
| Echo | `ivs_measurement`, `lvpw_measurement`, `pasp_value`, `tr_max_velocity_value`, `lvef_value` | Numeric |
| Demographics | `age_at_ecg` | Numeric |
| Demographics | `sex` | Categorical |

## Data Splits (all versions)

| Split | Normal | Mild | Moderate | Severe | Total | Note |
|---|---|---|---|---|---|---|
| Train | 3,000 | 1,974 | 1,339 | 1,580 | 7,893 | Normal undersampled to 3,000 |
| Val | 3,351 | 136 | 101 | 151 | 3,739 | Untouched — real-world prevalence |
| Test | 3,999 | 151 | 114 | 172 | 4,436 | Untouched — real-world prevalence |
| No-split | — | — | — | — | 15,114 | Excluded (unknown split origin) |

---

## Version Comparison

### Configuration

| Parameter | model_v1 | model_v2 |
|---|---|---|
| `embedding_dim` | 32 | 32 |
| `num_transformer_blocks` | 2 | 2 |
| `num_heads` | 4 | 4 |
| `dense_dim` | 32 | 32 |
| `dropout_rate` | 0.1 | **0.3** |
| Training shuffle | ✗ (bug) | **✓** |
| Best epoch | 100 (no early stop) | **7** |
| Epochs run | 100 | 17 |

### Results — Test Set

| Metric | model_v1 | model_v2 | model_v3 |
|---|---|---|---|
| Accuracy | 0.5780 | **0.6844** | 0.6758 |
| Macro F1 | 0.2837 | 0.3044 | **0.3186** |
| Weighted F1 | 0.6867 | **0.7590** | 0.7553 |
| Macro AUROC | 0.7364 | **0.7774** | 0.7757 |
| Macro AUPRC | 0.3115 | **0.3327** | 0.3248 |

### Results — Validation Set

| Metric | model_v1 | model_v2 | model_v3 |
|---|---|---|---|
| Accuracy | 0.5764 | **0.6737** | 0.6654 |
| Macro F1 | 0.2986 | 0.2994 | **0.3121** |
| Weighted F1 | 0.6811 | **0.7483** | 0.7447 |
| Macro AUROC | 0.7362 | 0.7611 | **0.7674** |
| Macro AUPRC | 0.3255 | **0.3371** | 0.3306 |

---

## Key Findings Per Version

**model_v1** — Baseline  
Training ran all 100 epochs with no early stopping. Train/val loss diverged from ~epoch 40 (overfitting). Low macro F1 despite decent AUROC due to model bias toward the dominant `normal` class. Silent shuffle bug meant batches were always in the same order.

**model_v2** — Shuffle fix + Dropout 0.3  
Early stopping triggered at epoch 7 (stopped at 17). All metrics improved, accuracy +0.10. Shuffle fix had the single largest impact. Macro F1 still low — minority class recall remains the bottleneck.

**model_v3** — SMOTE on minority classes  
SMOTENC balanced all 4 classes to 3,000 each (12,000 total training samples). Best epoch was 4 (stopped at 14) — model converged faster. Macro F1 improved to 0.3186 (best across all versions), confirming the model is now predicting minority classes more often. Accuracy and AUROC slightly lower than v2 — expected trade-off: better minority recall at the cost of slightly more normal class false positives.

---

## Ablation Roadmap

| Version | Key change | Status |
|---|---|---|
| model_v1 | Baseline | ✅ Done |
| model_v2 | Shuffle fix + dropout 0.3 | ✅ Done |
| model_v3 | SMOTENC on minority classes | ✅ Done |
| model_v4 | — | 🔲 Planned |
