# AutoML vs Linear Probing — Progress Notes

_Last updated: 2026-09-22 10:50 MDT_

## Goal

Test whether AutoGluon (AutoML: HPO + ensembling) beats a standard linear
probe when both are trained on the same Perch V2 embeddings and the same
train/eval split, on a real subset of WABAD. Question: does the AutoML
advantage hold across classes and across training-set sizes?

## Status: experiment running

## Setup decisions (and why)

- **Environment**: fresh venv at `automl/.venv`. Installed: pandas, numpy,
  scikit-learn, onnxruntime, librosa, soundfile, autogluon.tabular,
  lightgbm/xgboost/catboost/torch (+ `brew install libomp`, required on
  macOS for lightgbm/xgboost to import at all). fastai still fails to
  import (skipped by AutoGluon automatically — not fatal, just one fewer
  model family in the ensemble).
- **Embedding model**: `sammlapp/Perch_v2_headless` (ONNX, embedding-only,
  no classification head, ~94MB). Input: raw waveform `[batch, 160000]`
  (5s @ 32kHz). Output: `[batch, 1536]` embedding. Chosen over the PyTorch
  Perch V2 checkpoint because it's small, dependency-light (just
  onnxruntime), and matches "Perch V2" from the README.
- **Dataset**: WABAD (`DBD-research-group/WABAD` on HF), 72 sites total.
  Downloaded 8 sites (~3.6GB): ARD, BAM, BERB, HAG, KIB, SAL, BIAL, BMT —
  picked for geographic/biome diversity and reasonable size, not all 72
  (would be tens of GB, infeasible for this session).
- **Task framing**: WABAD annotations are multi-label per clip. For a clean
  multi-class comparison (matching typical few-shot transfer-learning
  benchmark setup, e.g. Ghani et al.), we restrict to **single-label clips
  only** and keep species with >=40 such clips, pooled across the 8 sites.
  Result: **19 species, 1456 train clips / 368 test clips** (native WABAD
  train/test split preserved, not re-split ourselves).
- **Audio -> embedding**: clips resampled to 32kHz, split into
  non-overlapping 5s windows (zero-padded if shorter), embedded per window,
  averaged to one 1536-dim vector per clip.
- **Experiment**: for training fractions [0.1, 0.25, 0.5, 1.0] (stratified
  per-class subsample of the 1456-clip train pool, 3 seeds per fraction
  except 1.0 which is deterministic), fit both:
  - Linear probe: `LogisticRegressionCV` (10 C values, balanced class
    weights, standardized features) — the standard baseline.
  - AutoGluon `TabularPredictor` (`presets='best_quality'`,
    `eval_metric='f1_macro'`, `time_limit=180s` per fit).
  Both evaluated on the same fixed 368-clip test set. Recording accuracy,
  macro-F1, and per-class F1 for both.

## Known issues hit & fixed

1. **libomp missing** — lightgbm/xgboost couldn't `dlopen` on macOS.
   Fixed with `brew install libomp`.
2. **Zip layout inconsistent across WABAD sites** — ARD extracts to
   `data_audio/ARD/audio/*.wav`, other sites extract flat to
   `data_audio/<site>/*.wav`. Fixed manifest builder to resolve whichever
   path actually exists on disk.
3. **`LogisticRegressionCV(multi_class=...)` removed in sklearn 1.9** —
   dropped the kwarg; multinomial is automatic for multi-class + lbfgs now.
4. **Background process management**: `nohup cmd & disown` inside this
   sandboxed Bash tool gets silently killed once the tool call returns
   (no error, no exit message — just gone, orphaned worker processes left
   behind). Fix: always launch long-running work directly via the Bash
   tool's own `run_in_background: true`, not manual `nohup`/`disown`.

## Results so far

_(pending — experiment in progress)_

## Next steps

1. Let `scripts/03_run_experiment.py` finish (writes
   `results/raw_results.json` incrementally after each frac/seed combo).
2. Run `scripts/04_analyze_results.py` to produce summary tables
   (`results/summary_aggregated.csv`, `results/delta_macro_f1.csv`,
   `results/delta_accuracy.csv`, per-class F1 deltas at the largest
   fraction).
3. Write up findings: does AutoGluon beat the linear probe overall? At
   which training fractions is the gap largest (expect: bigger AutoML
   edge at larger n, since AutoGluon needs enough data to do meaningful
   HPO/ensembling; less clear-cut in the very-low-shot regime)? Which
   species benefit most/least?
4. Commit + push periodically (data/models/audio excluded via
   `.gitignore`; only code, scripts, and small result CSVs/JSON tracked).

## Repo layout

```
scripts/01_build_manifest.py       # WABAD metadata -> single-label multiclass manifest
scripts/02_extract_embeddings.py   # Perch V2 ONNX embeddings for manifest clips
scripts/03_run_experiment.py       # linear probe vs AutoGluon, multiple train fractions/seeds
scripts/04_analyze_results.py      # aggregate raw_results.json -> summary tables
data/                              # WABAD metadata + audio zips (gitignored)
data_audio/                        # extracted WABAD audio (gitignored)
models/                            # Perch V2 ONNX + labels (gitignored, re-downloadable)
results/                           # raw_results.json + summary CSVs (tracked, small)
```
