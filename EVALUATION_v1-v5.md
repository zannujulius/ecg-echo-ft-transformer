# FT-Transformer Codebase Evaluation (model_v1 → model_v5)

**Task:** 4-class Aortic Stenosis (AS) severity — `normal · mild · moderate · severe` on EchoNext.
**Reviewed:** all 5 READMEs, all 5 `model.ipynb` (architecture, data pipeline, training, eval), training-history / confusion-matrix plots, and the raw CSV.
**Scope:** evaluation only — no code changed.

---

## 1. What the ablation chain actually shows

| Version | Change | Test Acc | Macro F1 | Macro AUROC |
|---|---|---|---|---|
| v1 | baseline (shuffle **bug**, dropout 0.1) | 0.578 | 0.284 | 0.736 |
| v2 | shuffle fix + dropout 0.3 | 0.684 | 0.304 | 0.777 |
| v3 | SMOTENC minority → 3k each | 0.676 | 0.319 | 0.776 |
| v4 | normal cap 2k, all real data | 0.669 | 0.319 | **0.783** |
| v5 | focal loss γ=2 | 0.681 | **0.328** | 0.764 |

The single real win is **v1 → v2** (shuffle fix). Everything after that moves Macro F1 by ±0.01 — which, as Section 3 shows, is inside the run-to-run noise. The project has effectively plateaued at Macro F1 ≈ 0.30–0.33 and the READMEs are over-interpreting noise as signal.

---

## 2. What was done *right* (keep these)

- **Splits are patient-disjoint.** I checked: 0 patients shared between train/val/test. The 3,627 overlapping patients are all in `no_split`, which is correctly excluded. No leakage. (Good — this is the most common mistake in EchoNext-style data and you avoided it.)
- **Preprocessing pipeline is fit on train only** (median impute + StandardScaler), then applied to val/test. Correct, no leakage.
- **Val/test kept at natural prevalence** — honest, real-world evaluation rather than a rebalanced test set.
- **The v1 shuffle bug was real and correctly diagnosed** — `tf.data` without `.shuffle()` feeds identical batch order every epoch.
- **The FT-Transformer implementation is sound**: per-feature linear tokenizer for numerics, StringLookup+Embedding for `sex`, standard pre-LN-ish transformer blocks.

---

## 3. The biggest problems (in priority order)

### 3.1 Model selection is driven by the wrong metric, on a noisy signal — **the core methodological flaw**
- Early stopping and `restore_best_weights` monitor **`val_loss`**, computed on the natural (≈90% normal) val set. That loss is dominated by the easy `normal` class, so you are **selecting the checkpoint that is best at `normal`, not best at minority detection** — the opposite of your stated objective (Macro F1).
- The val curves (v2, v5) are violently jagged: val loss swings ~0.47↔0.57 and val accuracy ~0.58↔0.70 **every epoch**. "Best epoch 4" / "best epoch 7" is being picked out of noise. Re-running with a different seed would pick a different epoch and different metrics.
- **Consequence:** the v2→v3→v4→v5 deltas (±0.01 Macro F1) are not trustworthy. They are single-seed point estimates with no error bars.
- **Fix:** (a) monitor a minority-sensitive metric for early stopping/checkpointing — macro-F1 or macro-AUPRC via a custom callback, not val_loss; (b) report **mean ± std over ≥5 seeds** before claiming any ablation "won."

### 3.2 The label is treated as nominal, but AS severity is **ordinal** → "moderate" collapses
- v5 confusion matrix: `moderate` recall is **5.3%** (6 of 114). The model has essentially deleted the moderate class; it gets absorbed into mild/severe (adjacent levels). Severe precision is ~21% (102 correct out of ~488 predicted).
- Softmax + CE/focal treats normal↔severe the same as moderate↔severe. The errors are overwhelmingly **adjacent-class** confusions — the signature of forcing an ordinal target through a nominal head.
- **Fix options:** ordinal regression head (CORN/CORAL), or report **macro AUPRC / quadratic-weighted kappa** as the headline metric instead of Macro F1, which is brutal on a tiny middle class.

### 3.3 Focal loss **and** class weights are stacked — likely over-correcting
- v5 applies `class_weight=balanced` **and** focal loss simultaneously. The README calls this the "alpha-balanced variant," but the focal implementation has **no alpha term** — the balancing is entirely Keras `class_weight`. So minority classes are up-weighted by `class_weight` *and* by the focal modulation. This is double pressure toward minority classes and is exactly why `normal` recall drops to 71% (796 normals predicted as mild) and minority precision is poor.
- **Fix:** pick **one** rebalancing mechanism. Either focal-with-alpha and drop `class_weight`, or CE+`class_weight`. Then tune γ ∈ {1,2,3} and the threshold, rather than stacking.

### 3.4 The biggest ceiling is likely the **features**, not the model — verify before more tuning
- AS severity is clinically defined by **aortic-valve peak velocity, mean transvalvular gradient, and aortic valve area (AVA)**. **None** of those are in the input set. Your "echo" inputs are `ivs`, `lvpw`, `pasp`, `tr_max_velocity` (tricuspid, not aortic), `lvef` — all only weakly related to AS. The 5 ECG intervals carry little AS signal.
- Macro AUROC stuck at ~0.78 across every architecture/loss change is the fingerprint of a **feature-limited** problem, not a model-limited one. No amount of dropout/SMOTE/focal tuning will move a ceiling set by input informativeness.
- **Action:** run a quick gradient-boosted baseline (XGBoost/LightGBM) on the same 12 features. If FT-Transformer ≈ GBDT (likely), the bottleneck is the features, and the next move is more informative inputs (the raw ECG waveform, or true AV-gradient features) — not more transformer ablations.

### 3.5 The task framing may be harder than the dataset was designed for
- EchoNext ships an explicit binary label `aortic_stenosis_moderate_or_greater_flag` — that "significant-AS screening" task is what the dataset was built for. Your 4-class split forces the model to separate `mild` vs `moderate` vs `severe` on tiny supports (114–172 test cases each) with features that barely distinguish them.
- **Action:** run the **binary `moderate-or-greater` task** as a reference point. It is the clinically meaningful endpoint, the metrics will be far more stable, and it tells you how much of the poor Macro F1 is task difficulty vs model.

---

## 4. Smaller issues / cleanups

- **No fixed global seed** for TF/NumPy weight init — combined with single runs, results aren't reproducible run-to-run. Set `tf.random.set_seed` / `np.random.seed` and report seeds.
- **`restore_best_weights` on a noisy val_loss** can restore a fluke epoch — same root cause as 3.1.
- **No threshold tuning.** Everything uses `argmax`. With heavy class weighting, per-class thresholds (or `argmax` over calibrated probabilities) would recover a lot of the lost `normal` precision.
- **No probability calibration** reported, yet you interpret AUROC/AUPRC drops as "calibration shifts." If calibration matters clinically, measure it (reliability curve / ECE).
- **`[CLS]` token not used** — you pool with `GlobalAveragePooling1D`. Fine, but the canonical FT-Transformer uses a learnable CLS token; worth trying as a cheap ablation.
- **Architecture capacity was never actually scaled.** Every version keeps `embedding_dim=32, blocks=2, heads=4` (13.8k params). The v1 README lists "increase embedding_dim/blocks/heads" as ideas but no version tested them. The ablations only touched data/loss, never the transformer itself — so "best FT-Transformer config" is still unknown.

---

## 5. Recommended next steps (highest leverage first)

1. **Establish a noise floor.** Re-run v4 and v5 with 5 seeds each; report mean ± std. Confirm whether any post-v2 change is real. *(One afternoon; reframes everything.)*
2. **Fix model selection.** Early-stop / checkpoint on **macro-AUPRC** (or macro-F1), not val_loss.
3. **Add a GBDT baseline** (XGBoost/LightGBM) on the same features → diagnoses whether the ceiling is features or model.
4. **Run the binary moderate-or-greater task** as the clinically-aligned reference.
5. **Stop stacking rebalancers** — choose focal-with-alpha *or* class_weight, then tune γ and thresholds.
6. **Try ordinal modeling** (CORN head) or switch the headline metric to quadratic-weighted kappa, given the ordinal target and moderate-class collapse.
7. **Only then** do the planned v6 Optuna search — and search the *architecture* (embedding_dim, blocks, heads, LR), which has never been explored, not just the loss.

**Bottom line:** the pipeline is clean and leak-free, but the ablation conclusions past v2 are being read out of single-seed noise, model selection is optimizing the wrong thing, and the real ceiling is almost certainly the input features / 4-class framing — not the transformer. Validate the noise floor and the feature ceiling before investing in more architecture tuning.
