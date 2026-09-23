"""
All-species, small-training-size follow-up: linear probe vs AutoGluon on
EVERY species with enough data, at training-positive targets 5, 25, and 100
(per user request). One-vs-rest binary presence/absence per species, same
framing as scripts/07_run_multilabel_experiment.py (see scripts/05_build_multilabel_manifest.py
for why WABAD is treated as multi-label rather than single-label multiclass).

Reuses embeddings_all_species.npz (455 species, same 25-site clip pool as
the 8-species Phase 2 run -- embeddings reused, not re-extracted; see
scripts/10_build_all_species_embeddings.py).

Scale note: 455 species x up to 3 targets (a species only gets a target if
it has enough train positives) = ~900 species-target combos. Given the
scale, this runs with a SINGLE seed per combo (not 3, unlike the 8-species
Phase 2 run) to keep total runtime tractable (~1.5-2 hours). Single-seed
results at n=5-100 will have higher variance than the multi-seed Phase 2
numbers -- treat per-species results as noisier, and lean on the
aggregate (mAP / macro-AUROC across all species) for the headline claim.
"""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EMB_PATH = Path("/Users/SML161/automl/data/embeddings_all_species.npz")
RESULTS_DIR = Path("/Users/SML161/automl/results")
AG_MODEL_DIR = Path("/Users/SML161/automl/models/autogluon_all_species_tmp")

TRAIN_POS_TARGETS = [5, 25, 100]
NEG_POS_RATIO = 3
SEED = 0


def load_data():
    d = np.load(EMB_PATH, allow_pickle=True)
    mask = d["valid_mask"]
    X = d["embeddings"][mask]
    labels = d["labels"][mask]
    label_names = [n.replace("label__", "").replace("_", " ") for n in d["label_names"]]
    split = d["split"][mask]
    return X, labels, label_names, split


def subsample_train(pos_idx, neg_idx, target_pos, neg_ratio, seed):
    rng = np.random.RandomState(seed)
    n_pos = min(target_pos, len(pos_idx))
    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    n_neg = min(n_pos * neg_ratio, len(neg_idx))
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)
    return idx, n_pos, n_neg


def binary_metrics(y_true, y_pred, y_score):
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_score),
        "auroc": auroc,
    }


def fit_eval_linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    min_class_count = pd.Series(y_train).value_counts().min()
    cv_folds = max(2, min(5, min_class_count))

    clf = LogisticRegressionCV(
        Cs=10,
        cv=cv_folds,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
        scoring="average_precision",
    )
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    scores = clf.predict_proba(X_test_s)[:, 1]
    return binary_metrics(y_test, preds, scores)


def fit_eval_autogluon(X_train, y_train, X_test, y_test, tag):
    from autogluon.tabular import TabularPredictor
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    feat_cols = [f"emb_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["label"] = y_train
    test_df = pd.DataFrame(X_test, columns=feat_cols)

    save_path = AG_MODEL_DIR / tag
    if save_path.exists():
        shutil.rmtree(save_path)

    # Same stable model-family restriction as scripts/03 and 07 (GBM/CAT/XGB/FASTAI
    # are unusable on this machine -- see PROGRESS.md "Environment issues").
    hyperparameters = get_hyperparameter_config("default")
    for family in ("GBM", "CAT", "XGB", "FASTAI"):
        hyperparameters.pop(family, None)

    predictor = TabularPredictor(
        label="label",
        path=str(save_path),
        eval_metric="average_precision",
        verbosity=0,
    ).fit(
        train_df,
        hyperparameters=hyperparameters,
        presets="medium_quality",
        time_limit=30,  # small data -- fits finish in seconds regardless
    )

    preds = predictor.predict(test_df).values.astype(int)
    scores = predictor.predict_proba(test_df)[1].values
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]["model"] if len(leaderboard) else None

    shutil.rmtree(save_path, ignore_errors=True)

    metrics = binary_metrics(y_test, preds, scores)
    metrics["best_model"] = best_model
    return metrics


def main():
    X, labels, label_names, split = load_data()
    train_mask = split == "train"
    test_mask = split == "test"
    X_test_all = X[test_mask]

    print(f"Total clips: {len(X)} (train={train_mask.sum()}, test={test_mask.sum()})")
    print(f"Species: {len(label_names)}")

    AG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "raw_results_all_species.json"
    all_results = []
    done_tags = set()
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_tags = {f"{r['species']}_pos{r['target_pos']}" for r in all_results}
        print(f"Resuming: {len(done_tags)} combos already done")

    for sp_idx, species in enumerate(label_names):
        y_all = labels[:, sp_idx]
        y_train_all = y_all[train_mask]
        y_test = y_all[test_mask]

        pos_idx_all = np.where(train_mask)[0][y_train_all == 1]
        neg_idx_all = np.where(train_mask)[0][y_train_all == 0]
        n_test_pos = int(y_test.sum())

        for target_pos in TRAIN_POS_TARGETS:
            if len(pos_idx_all) < target_pos:
                continue  # not enough positives for this target

            tag = f"{species}_pos{target_pos}"
            if tag in done_tags:
                continue

            sub_idx, n_pos, n_neg = subsample_train(pos_idx_all, neg_idx_all, target_pos, NEG_POS_RATIO, SEED)
            X_train, y_train = X[sub_idx], y_all[sub_idx]

            print(f"=== [{sp_idx+1}/{len(label_names)}] {tag}: n_train={len(sub_idx)} (pos={n_pos}, neg={n_neg}), test_pos={n_test_pos} ===")

            try:
                lp_result = fit_eval_linear_probe(X_train, y_train, X_test_all, y_test)
                ag_result = fit_eval_autogluon(X_train, y_train, X_test_all, y_test, tag.replace(" ", "_"))
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

            print(f"  LP: AP={lp_result['average_precision']:.3f} AUROC={lp_result['auroc']:.3f} | "
                  f"AG: AP={ag_result['average_precision']:.3f} AUROC={ag_result['auroc']:.3f} ({ag_result['best_model']})")

            all_results.append({
                "species": species,
                "target_pos": target_pos,
                "n_train": len(sub_idx),
                "n_train_pos": n_pos,
                "n_train_neg": n_neg,
                "n_test_pos": n_test_pos,
                "n_test": len(y_test),
                "linear_probe": lp_result,
                "autogluon": ag_result,
            })

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print("\nDone. Results saved to results/raw_results_all_species.json")


if __name__ == "__main__":
    main()
