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

| Metric | model_v1 | model_v2 | model_v3 | model_v4 | model_v5 |
|---|---|---|---|---|---|
| Accuracy | 0.5780 | **0.6844** | 0.6758 | 0.6693 | 0.6812 |
| Macro F1 | 0.2837 | 0.3044 | 0.3186 | 0.3188 | **0.3283** |
| Weighted F1 | 0.6867 | **0.7590** | 0.7553 | 0.7517 | 0.7588 |
| Macro AUROC | 0.7364 | 0.7774 | 0.7757 | **0.7830** | 0.7635 |
| Macro AUPRC | 0.3115 | 0.3327 | 0.3248 | **0.3353** | 0.3346 |

### Results — Validation Set

| Metric | model_v1 | model_v2 | model_v3 | model_v4 | model_v5 |
|---|---|---|---|---|---|
| Accuracy | 0.5764 | **0.6737** | 0.6654 | 0.6579 | 0.6676 |
| Macro F1 | 0.2986 | 0.2994 | 0.3121 | 0.3107 | **0.3213** |
| Weighted F1 | 0.6811 | **0.7483** | 0.7447 | 0.7408 | 0.7472 |
| Macro AUROC | 0.7362 | 0.7611 | 0.7674 | **0.7686** | 0.7532 |
| Macro AUPRC | 0.3255 | 0.3371 | 0.3306 | 0.3349 | **0.3371** |

---

## Key Findings Per Version

**model_v1** — Baseline  
Training ran all 100 epochs with no early stopping. Train/val loss diverged from ~epoch 40 (overfitting). Low macro F1 despite decent AUROC due to model bias toward the dominant `normal` class. Silent shuffle bug meant batches were always in the same order.

**model_v2** — Shuffle fix + Dropout 0.3  
Early stopping triggered at epoch 7 (stopped at 17). All metrics improved, accuracy +0.10. Shuffle fix had the single largest impact. Macro F1 still low — minority class recall remains the bottleneck.

**model_v3** — SMOTE on minority classes  
SMOTENC balanced all 4 classes to 3,000 each (12,000 total). Macro F1 improved to 0.3186. AUROC/AUPRC slightly lower than v2 — SMOTE synthetic data added noise.

**model_v4** — Normal cap 2,000, no SMOTE (all real data)  
Reducing normal to 2,000 gives a near-balanced training set (29/29/23/19%) using only 6,893 real patients. Macro F1 matches v3 (0.3188 ≈ same) with better AUROC (0.7830, best so far) and AUPRC (0.3353). Confirms that real data quality beats synthetic quantity. Best baseline for v5 focal loss.

---

## Configuration Summary

| Parameter | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Normal cap | 3,000 | 3,000 | 3,000 | **2,000** | 2,000 |
| SMOTE | ✗ | ✗ | ✓ (all→3k) | ✗ | ✗ |
| Training samples | 7,893 | 7,893 | 12,000 | 6,893 | 6,893 |
| Shuffle | ✗ | ✓ | ✓ | ✓ | ✓ |
| Dropout | 0.1 | 0.3 | 0.3 | 0.3 | 0.3 |
| Loss | CE | CE | CE | CE | **Focal (γ=2)** |
| Best epoch | 100 | 7 | 4 | 4 | 4 |

---

## Ablation Roadmap

| Version | Key change | Status |
|---|---|---|
| model_v1 | Baseline | ✅ Done |
| model_v2 | Shuffle fix + dropout 0.3 | ✅ Done |
| model_v3 | SMOTENC on minority classes | ✅ Done |
| model_v4 | Normal cap 2,000 — all real data | ✅ Done |
| model_v5 | Focal loss (γ=2) on v4 setup | ✅ Done |
| model_v6 | Hyperparameter search (Optuna) on v5 setup | 🔲 Planned |
