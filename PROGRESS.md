# AutoML vs Linear Probing — Progress Notes

_Last updated: 2026-09-22 12:05 MDT_

## Goal

Test whether AutoGluon (AutoML: HPO + ensembling) beats a standard linear
probe when both are trained on the same Perch V2 embeddings and the same
train/eval split, on a real subset of WABAD. Question: does the AutoML
advantage hold across classes and across training-set sizes?

## Status: experiment complete, results in

## Setup

- **Environment**: venv at `automl/.venv`. pandas, numpy, scikit-learn,
  onnxruntime, librosa, soundfile, autogluon.tabular, torch (CPU),
  lightgbm/xgboost/catboost installed but ultimately excluded from the
  AutoGluon run — see "Environment issues" below. Required
  `brew install libomp` for lightgbm/xgboost to even import on macOS.
- **Embedding model**: `sammlapp/Perch_v2_headless` (ONNX, embedding-only,
  ~94MB). Input: raw waveform `[batch, 160000]` (5s @ 32kHz). Output:
  `[batch, 1536]` embedding.
- **Dataset**: WABAD (`DBD-research-group/WABAD` on HF), 72 sites total.
  Downloaded 8 sites (~3.6GB): ARD, BAM, BERB, HAG, KIB, SAL, BIAL, BMT —
  picked for geographic/biome diversity, not all 72 (would be tens of GB).
- **Task framing**: WABAD annotations are multi-label per clip. For a clean
  multi-class comparison (matching typical few-shot transfer-learning
  benchmark setups, e.g. the Ghani et al. approach cited in the README), we
  restrict to **single-label clips only** and keep species with >=40 such
  clips, pooled across the 8 sites. Result: **19 species, 1456 train clips
  / 368 test clips** (native WABAD train/test split preserved).
- **Audio -> embedding**: clips resampled to 32kHz, split into
  non-overlapping 5s windows (zero-padded if shorter), embedded per window,
  averaged to one 1536-dim vector per clip.
- **Experiment**: for training fractions [0.1, 0.25, 0.5, 1.0] (stratified
  per-class subsample of the 1456-clip train pool; 3 seeds per fraction
  except 1.0, which is deterministic — 10 combos total), fit both:
  - **Linear probe**: `LogisticRegressionCV` (10 C values, balanced class
    weights, standardized features) — the standard transfer-learning
    baseline.
  - **AutoGluon** `TabularPredictor` (`presets='medium_quality'`,
    `eval_metric='f1_macro'`, `time_limit=120s`; model zoo: RandomForest
    x2, ExtraTrees x2, PyTorch NN, + weighted ensemble — see below for why).
  Both evaluated on the same fixed 368-clip test set.

## Environment issues hit & fixed (macOS arm64 specific)

1. **libomp missing** — lightgbm/xgboost couldn't `dlopen`. Fixed with
   `brew install libomp`.
2. **WABAD zip layout inconsistent across sites** — ARD extracts to
   `data_audio/ARD/audio/*.wav`, other sites extract flat to
   `data_audio/<site>/*.wav`. Manifest builder resolves whichever path
   actually exists on disk.
3. **`LogisticRegressionCV(multi_class=...)` removed in sklearn 1.9** —
   dropped the kwarg (multinomial is now automatic for multiclass+lbfgs).
4. **Background process handling**: `nohup cmd & disown` gets silently
   killed by this sandboxed environment once the launching tool call
   returns. Fixed by using the harness's own `run_in_background` /
   `Monitor` mechanisms, and making the experiment script resumable
   (skips frac/seed combos already in `results/raw_results.json`).
5. **Every native-code boosting library AutoGluon tried was unusable
   here**: LightGBM segfaults reliably inside AutoGluon's bagged fitting
   (reproduced in isolation; raw `lightgbm.train()` is fine standalone, so
   it's AutoGluon's wrapper/threading, independent of `OMP_NUM_THREADS`).
   XGBoost segfaults intermittently the same way (also fine standalone).
   CatBoost doesn't crash but is pathologically slow on 1536 raw embedding
   features (~0.4s/iteration) and eats the *entire* time budget alone,
   starving every other model family even at `time_limit=300s`.
   `presets='best_quality'` (8-fold bagging x 2 stack levels) also crashed
   intermittently, likely resource contention from nested multiprocessing
   given how small the per-class sample counts get (as few as 3 per class
   at the smallest fraction).
   **Fix**: `presets='medium_quality'` (no bagging/stacking) with only
   `RF`, `XT` (ExtraTrees), `NN_TORCH` in the hyperparameter search — GBM/
   CAT/XGB/FASTAI excluded. Verified stable: 5 base models + 1 weighted
   ensemble, ~4s/fit. This is a narrower model zoo than AutoGluon's full
   default, but still genuinely AutoML (multiple families, hyperparameter
   variants, learned ensembling) — just missing the gradient-boosting
   families that are broken in this specific environment. **Caveat**: a
   fairer real-world AutoGluon run (Linux, or a working GBM stack) would
   likely include LightGBM/XGBoost/CatBoost and could score higher; this
   result reflects AutoGluon's tree/NN ensemble, not its full model zoo.

## Results

**Bottom line: the linear probe beat AutoGluon at every training fraction
tested, and the gap shrinks as training data grows but doesn't close.**

Macro-F1, mean over seeds:

| train frac | n_train | linear probe | AutoGluon | delta (AG − LP) |
|---:|---:|---:|---:|---:|
| 0.10 | 144  | 0.779 | 0.681 | **−0.098** |
| 0.25 | 363  | 0.859 | 0.796 | **−0.063** |
| 0.50 | 728  | 0.887 | 0.831 | **−0.056** |
| 1.00 | 1456 | 0.907 | 0.888 | **−0.019** |

Accuracy shows the same pattern (e.g. 0.921 vs 0.905 at frac=1.0). Full
per-seed numbers: `results/summary_long.csv`; aggregated:
`results/summary_aggregated.csv`, `results/delta_macro_f1.csv`,
`results/delta_accuracy.csv`.

**Per-class, at the full training set (frac=1.0)**: linear probe wins on
8/19 species, AutoGluon wins on 5/19, and they tie on 6/19
(`results/per_class_f1_frac1.0.csv`). AutoGluon's biggest wins are on
*Oreolais pulcher* (+0.087 F1) and *Apalis cinerea* (+0.057); its biggest
losses are on *Cinnyris bouvieri* (−0.096) and *Luscinia megarhynchos*
(−0.091). No obvious pattern (e.g. by class frequency) separates the
winners from the losers — would need a larger species set to say more.

**Interpretation**: on high-dimensional (1536-d), small-sample embeddings,
a well-regularized linear probe is a strong baseline that's hard to beat
with tree ensembles — this matches known ML folklore (linear models on
deep embeddings tend to be competitive/superior to trees in the small-n,
large-p regime) but runs counter to the README's hypothesis. The result
here is *AutoML (as configured, with GBM/CatBoost/XGBoost excluded for
environment reasons) does not outperform linear probing* on this WABAD
subset with Perch V2 embeddings — the opposite of what was expected. The
gap narrowing as n grows is consistent with AutoGluon's ensemble needing
more data to add value over a single well-tuned linear model, but even at
n=1456 (77 clips/class average) it hasn't caught up. A version of this
experiment with a working gradient-boosting stack (e.g. on Linux) might
close some of this gap; that's the most important caveat on these numbers.

## Repo layout

```
scripts/01_build_manifest.py       # WABAD metadata -> single-label multiclass manifest
scripts/02_extract_embeddings.py   # Perch V2 ONNX embeddings for manifest clips
scripts/03_run_experiment.py       # linear probe vs AutoGluon, multiple train fractions/seeds
scripts/04_analyze_results.py      # aggregate raw_results.json -> summary tables
data/                              # WABAD metadata + audio zips (gitignored, re-fetchable)
data_audio/                        # extracted WABAD audio (gitignored, re-fetchable)
models/                            # Perch V2 ONNX + labels (gitignored, re-downloadable)
results/                           # raw_results.json + summary CSVs (tracked, small)
```

## Possible follow-ups (not done)

- Re-run with a working LightGBM/XGBoost/CatBoost stack (e.g. Linux box or
  Docker) to see whether AutoGluon's full model zoo closes the gap.
- Expand beyond 8 WABAD sites / 19 species for more statistical power on
  the per-class breakdown.
- Try AutoGluon's `extreme_quality`/foundation-model presets (TabPFN etc.)
  now that `noncommercial_2026_08_05`-style portfolios exist — may be
  better suited to small-n tabular data than tree ensembles.
