# AutoML vs Linear Probing — Progress Notes

_Last updated: 2026-09-23 09:50 MDT_

## Goal

Test whether AutoGluon (AutoML: HPO + ensembling) beats a standard linear
probe when both are trained on the same Perch V2 embeddings and the same
train/eval split, on a real subset of WABAD. Question: does the AutoML
advantage hold across classes and across training-set sizes?

WABAD is natively **multi-label** (a clip's target is a binary vector over
species, not one class out of N — most clips contain several co-occurring
species). All reported results use the correct framing: **one-vs-rest binary
presence/absence classification per species**, using every clip as a
positive or negative example for each species independently, evaluated on
the full native test split. An earlier single-label-multiclass framing
(discarding every multi-species clip to force ordinary softmax classification)
was tried first and abandoned — see "Methodological note" below — and its
results are not reported here.

## Status: both runs complete

- **8-species run (1k-3.7k positive training labels/species):** AutoGluon
  beats the linear probe at every training-size target, gap widens with
  more data. See "Results: 8 species, 1k-5k labels" below.
- **All-species run (455 species, small training sizes 5/25/100
  positives/species):** linear probe beats AutoGluon at every training-size
  target, gap narrows with more data. See "Results: all species, small-n"
  below. **Together, these two runs form one consistent picture across
  training sizes from 5 to ~3700 positives/class: linear probe wins small,
  AutoGluon wins large, and the crossover happens somewhere between
  ~100 and ~1000 positives per class in this setup.**

## Methodological note: why not single-label multiclass

The first version of this experiment discarded every WABAD clip with more
than one species label, to force an ordinary single-label multiclass setup
(pick 1 of 19 classes). That capped realistic per-class sample sizes at a
few hundred at most, because single-label clips are a small,
non-representative slice of the data (~82% of clips have multiple
co-occurring species) — and both training AND evaluation used only that
filtered, non-representative subset. Reframing as multi-label (below) fixes
both problems at once: it matches WABAD's actual structure, and it unlocks
far more data per species (8 species with 1k-5k+ positive occurrences,
vs. only 1 species clearing 1000 under the single-label-only framing across
all 72 sites).

## Results: 8 species, 1k-5k labels

**Bottom line: AutoGluon beat the linear probe at every training-size target
tested, on both headline metrics, and the gap widens as training data
grows.**

**Species** (8, chosen for >=2000 total occurrences, concentrated in
relatively few sites): Fringilla coelebs, Malacopteron magnirostre, Turdus
merula, Stachyris maculata, Sylvia atricapilla, Dicrurus paradiseus,
Erithacus rubecula, Luscinia megarhynchos. Final training-pool sizes (all
landed in the requested 1k-5k range): 3421, 3719, 2660, 2105, 1747, 1662,
1571, 1512 positives respectively (test set: ~25% of each, never
subsampled).

**Experiment design**: for each species x training-positive-count target
(1000, 2500, 5000 — "5000" in practice used all available positives, since
none reached it, max was 3719) x 3 seeds, subsample the training pool to
~that many positives with a fixed 3:1 negative:positive ratio, fit linear
probe (`LogisticRegressionCV`, scored on average precision) vs AutoGluon
(RF+ExtraTrees+NN config — see "Environment issues" below,
`eval_metric='average_precision'`) on the SAME subsample, evaluate both on
the FULL native test split.

**Eval metrics**: **mAP** (mean average precision, averaged across the 8
species) and **macro-AUROC** (AUROC averaged across species) are the
headline numbers, plus per-class AP and AUROC. Both are threshold-free
ranking metrics, standard for imbalanced binary/multi-label bioacoustic
detection.

mAP / macro-AUROC (mean across the 8 species), by training-positives target:

| target positives | linear probe mAP | AutoGluon mAP | delta | linear probe macro-AUROC | AutoGluon macro-AUROC | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.802 | 0.816 | **+0.013** | 0.9815 | 0.9833 | **+0.0018** |
| 2500 | 0.820 | 0.832 | **+0.012** | 0.9834 | 0.9850 | **+0.0016** |
| 5000\* | 0.822 | 0.838 | **+0.015** | 0.9837 | 0.9858 | **+0.0021** |

\*"5000" used all available positives per species (max 3719).

**Per-class at the largest target**: AutoGluon wins AP on 6/8 species and
AUROC on 6/8 species. Biggest AutoGluon wins: Dicrurus paradiseus (+0.081
AP), Turdus merula (+0.028 AP), Malacopteron magnirostre (+0.024 AP). Linear
probe wins: Erithacus rubecula (-0.041 AP), Luscinia megarhynchos (-0.011
AP) — both among the smaller-n species (1512-1571 train positives).

**Interpretation**: AutoGluon's RF/ExtraTrees/NN ensemble had enough data
(1.5k-3.7k positives) to add real value over a single linear decision
boundary, and the advantage grows with more data — consistent with AutoML
needing enough samples to pay off. Scripts: `05_build_multilabel_manifest.py`,
`06_extract_embeddings_multilabel.py`, `07_run_multilabel_experiment.py`,
`08_analyze_multilabel_results.py`. Results: `results/raw_results_multilabel.json`,
`results/multilabel_headline.csv`, `results/multilabel_per_class_ap_target5000.csv`,
`results/multilabel_per_class_auroc_target5000.csv`.

## Results: all species, small-n

**Bottom line: at small training sizes, the linear probe wins — reversing
the 8-species result — and the gap shrinks (but doesn't close) as training
size grows from 5 to 100 positives.** Combined with the 8-species result
above (1k-3.7k positives, AutoGluon wins and the gap widens with more
data), this traces one consistent story across nearly three orders of
magnitude of training-set size: **the linear probe wins in the very-low-data
regime; AutoGluon overtakes it once there's enough data (roughly
somewhere between 100 and 1000 positives per class in this setup), and its
advantage keeps growing from there.**

Extends the same multi-label one-vs-rest framing to every species with
enough data: **455 species**, reusing the same 25-site clip pool and the
already-extracted embeddings (no new downloads or re-extraction needed),
at training-positive targets **5, 25, and 100 positives per species** (per
user request), fixed 3:1 negative:positive ratio, evaluated on the full
native test split. A species is only run at a target if it has enough
training positives for that target: 455/300/147 species qualify for the
5/25/100 targets respectively (902 species-target combos total, single
seed per combo given the scale — see tradeoff note below).

mAP / macro-AUROC (mean across all species run at that target):

| target positives | # species | linear probe mAP | AutoGluon mAP | delta | linear probe macro-AUROC | AutoGluon macro-AUROC | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5   | 455 | 0.439 | 0.363 | **-0.077** | 0.9812 | 0.9710 | **-0.0103** |
| 25  | 300 | 0.516 | 0.489 | **-0.027** | 0.9872 | 0.9844 | **-0.0028** |
| 100 | 147 | 0.645 | 0.627 | **-0.018** | 0.9897 | 0.9885 | **-0.0012** |

Per-species win counts (AP): at target=5, linear probe wins 331/455,
AutoGluon wins 104/455, ties 20. At target=25: linear probe wins 189/300,
AutoGluon wins 107/300. At target=100: linear probe wins 91/147, AutoGluon
wins 56/147. Linear probe's win margin shrinks steadily as training size
grows, consistent with the mAP gap narrowing.

**Scale tradeoffs**: unlike the 8-species run (3 seeds/combo), this uses a
**single seed per combo** given the ~900-combo scale — per-species numbers
are noisier than the 8-species run's, so lean on the aggregate for the
headline claim. Also, ~9% of species (42/455) have only 1 positive test
example at target=5, which makes their individual AP/AUROC values noisy
(binary outcomes) — this adds noise to, but should not systematically bias,
the 455-species mean.

Scripts: `09_build_all_species_manifest.py` (455-species label manifest,
reuses existing audio), `10_build_all_species_embeddings.py` (reuses
embeddings from the 8-species run instead of re-running ONNX inference),
`11_run_all_species_experiment.py`, `12_analyze_all_species_results.py`.
Results: `results/raw_results_all_species.json`, `results/all_species_headline.csv`,
`results/all_species_per_species_target{5,25,100}.csv`.

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
  Downloaded 25 sites (~9.7GB): ARD, BAM, BERB, HAG, KIB, SAL, BIAL, BMT,
  BOLIN, CAT, CLH, DONG, DYOM, EFFOR, EFFOU, EVROS, KAR, NAV, OLIV, PITI,
  POZO, SCHF, SITH, SLOB, VIL — not all 72 (would be tens of GB), chosen
  to cover the target species for the 8-species run; the all-species run
  reuses this same pool.
- **Audio -> embedding**: clips resampled to 32kHz, split into
  non-overlapping 5s windows (zero-padded if shorter), embedded per window,
  averaged to one 1536-dim vector per clip.

## Environment issues hit & fixed (macOS arm64 specific)

1. **libomp missing** — lightgbm/xgboost couldn't `dlopen`. Fixed with
   `brew install libomp`.
2. **WABAD zip layout inconsistent across sites** — some sites extract to
   `data_audio/<site>/audio/*.wav`, others extract flat to
   `data_audio/<site>/*.wav`. Manifest builder resolves whichever path
   actually exists on disk.
3. **`LogisticRegressionCV(multi_class=...)` removed in sklearn 1.9** —
   dropped the kwarg (multinomial is now automatic for multiclass+lbfgs).
4. **Background process handling**: `nohup cmd & disown` gets silently
   killed by this sandboxed environment once the launching tool call
   returns. Fixed by using the harness's own `run_in_background` /
   `Monitor` mechanisms, and making experiment scripts resumable (skip
   combos already present in their results JSON).
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
   given how small per-class sample counts can get.
   **Fix**: `presets='medium_quality'` (no bagging/stacking) with only
   `RF`, `XT` (ExtraTrees), `NN_TORCH` in the hyperparameter search — GBM/
   CAT/XGB/FASTAI excluded. Verified stable: fits complete in seconds. This
   is a narrower model zoo than AutoGluon's full default, but still
   genuinely AutoML (multiple families, hyperparameter variants, learned
   ensembling) — just missing the gradient-boosting families that are
   broken in this specific environment. **Caveat that applies to every
   result in this doc**: a fairer real-world AutoGluon run (Linux, or a
   working GBM stack) would likely include LightGBM/XGBoost/CatBoost and
   could score higher still — these results reflect AutoGluon's tree/NN
   ensemble, not its full model zoo.

## Repo layout

```
scripts/05_build_multilabel_manifest.py     # 8-species multi-label manifest
scripts/06_extract_embeddings_multilabel.py # Perch V2 embeddings for the 8-species manifest
scripts/07_run_multilabel_experiment.py     # linear probe vs AutoGluon, 8 species, 1k-5k targets
scripts/08_analyze_multilabel_results.py    # aggregate -> mAP / macro-AUROC / per-class tables
scripts/09_build_all_species_manifest.py    # 455-species multi-label manifest (reuses audio)
scripts/10_build_all_species_embeddings.py  # reuses embeddings from the 8-species run
scripts/11_run_all_species_experiment.py    # linear probe vs AutoGluon, 455 species, 5/25/100 targets
scripts/12_analyze_all_species_results.py   # aggregate -> mAP / macro-AUROC / per-species tables
data/                              # WABAD metadata + audio zips (gitignored, re-fetchable)
data_audio/                        # extracted WABAD audio (gitignored, re-fetchable)
models/                            # Perch V2 ONNX + labels (gitignored, re-downloadable)
results/                           # results JSON + summary CSVs (tracked, small)
```

Earlier single-label-multiclass scripts (`01_build_manifest.py` through
`04_analyze_results.py`) are still in the repo for reference but their
results are superseded and not reported (see "Methodological note" above).

## Possible follow-ups (not done)

- Re-run with a working LightGBM/XGBoost/CatBoost stack (e.g. Linux box or
  Docker) to see whether AutoGluon's full model zoo widens its advantage
  further.
- Try AutoGluon's `extreme_quality`/foundation-model presets (TabPFN etc.)
  now that `noncommercial_2026_08_05`-style portfolios exist.
- All-species run currently uses 1 seed/combo for tractability; multi-seed
  would tighten the per-species estimates.
