"""
Compare a linear probe (logistic regression, the standard transfer-learning
baseline used in Perch/BirdSET-style papers) against AutoGluon (AutoML with
hyperparameter search + ensembling) trained on the SAME Perch V2 embeddings
and the SAME train/eval splits.

For each of several training-set fractions (subsampled per-class from the
dataset's native "train" split, stratified), we:
  1. Fit a linear probe (LogisticRegression, standardized features, light
     regularization sweep via cross-validation) on the sampled training set.
  2. Fit AutoGluon TabularPredictor (default presets, 'best_quality' where
     feasible) on the SAME sampled training set.
  3. Evaluate both on the SAME held-out "test" split (never subsampled).
  4. Record accuracy, macro-F1, and per-class F1 for both models.

Repeats each fraction with multiple random seeds for the subsampling to get
stable estimates given small per-class sample sizes.
"""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EMB_PATH = Path("/Users/SML161/automl/data/embeddings.npz")
RESULTS_DIR = Path("/Users/SML161/automl/results")
AG_MODEL_DIR = Path("/Users/SML161/automl/models/autogluon_tmp")

TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
MIN_PER_CLASS_TRAIN = 3  # skip fraction/seed combos that would leave a class empty


def load_data():
    d = np.load(EMB_PATH, allow_pickle=True)
    mask = d["valid_mask"]
    X = d["embeddings"][mask]
    y = d["species_name"][mask]
    split = d["split"][mask]
    return X, y, split


def subsample_train(y_train_idx, y_labels, frac, seed):
    """Stratified subsample of training indices by class, given a fraction."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"idx": y_train_idx, "label": y_labels})
    keep_idx = []
    for label, group in df.groupby("label"):
        n = len(group)
        k = max(1, int(round(n * frac)))
        chosen = rng.choice(group["idx"].values, size=k, replace=False)
        keep_idx.extend(chosen.tolist())
    return np.array(keep_idx)


def fit_eval_linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    min_class_count = pd.Series(y_train).value_counts().min()
    cv_folds = max(2, min(5, min_class_count))

    clf = LogisticRegressionCV(
        Cs=10,
        cv=cv_folds,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(X_train_s, y_train)
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    per_class_f1 = f1_score(y_test, preds, average=None, labels=sorted(np.unique(y_test)))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(sorted(np.unique(y_test)), per_class_f1.tolist())),
    }


def fit_eval_autogluon(X_train, y_train, X_test, y_test, tag):
    from autogluon.tabular import TabularPredictor

    feat_cols = [f"emb_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    train_df["label"] = y_train
    test_df = pd.DataFrame(X_test, columns=feat_cols)

    save_path = AG_MODEL_DIR / tag
    if save_path.exists():
        shutil.rmtree(save_path)

    predictor = TabularPredictor(
        label="label",
        path=str(save_path),
        eval_metric="f1_macro",
        verbosity=0,
    ).fit(
        train_df,
        presets="best_quality",
        time_limit=120,
    )

    preds = predictor.predict(test_df).values
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    per_class_f1 = f1_score(y_test, preds, average=None, labels=sorted(np.unique(y_test)))

    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]["model"] if len(leaderboard) else None

    # clean up disk (AutoGluon models can be large; we only need the metrics)
    shutil.rmtree(save_path, ignore_errors=True)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(sorted(np.unique(y_test)), per_class_f1.tolist())),
        "best_model": best_model,
        "n_models_trained": len(leaderboard),
    }


def main():
    X, y, split = load_data()
    train_idx_all = np.where(split == "train")[0]
    test_idx = np.where(split == "test")[0]

    X_test, y_test = X[test_idx], y[test_idx]
    y_train_all = y[train_idx_all]

    print(f"Total train pool: {len(train_idx_all)}, test: {len(test_idx)}")
    print(f"Classes: {len(np.unique(y))}")

    AG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "raw_results.json"
    all_results = []
    done_tags = set()
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_tags = {f"frac{r['frac']}_seed{r['seed']}" for r in all_results}
        print(f"Resuming: {len(done_tags)} combos already done: {sorted(done_tags)}")

    for frac in TRAIN_FRACTIONS:
        for seed in SEEDS:
            if frac == 1.0 and seed != SEEDS[0]:
                # frac=1.0 is deterministic (no subsampling), skip repeated seeds
                continue

            tag = f"frac{frac}_seed{seed}"
            if tag in done_tags:
                print(f"Skipping {tag}: already in raw_results.json")
                continue

            sub_idx = subsample_train(train_idx_all, y_train_all, frac, seed)
            X_train, y_train = X[sub_idx], y[sub_idx]

            class_counts = pd.Series(y_train).value_counts()
            if class_counts.min() < 1:
                print(f"Skipping frac={frac} seed={seed}: empty class")
                continue

            n_train = len(y_train)
            print(f"\n=== {tag}: n_train={n_train} (min/class={class_counts.min()}, max/class={class_counts.max()}) ===")

            lp_result = fit_eval_linear_probe(X_train, y_train, X_test, y_test)
            print(f"  Linear probe: acc={lp_result['accuracy']:.4f} macro_f1={lp_result['macro_f1']:.4f}")

            ag_result = fit_eval_autogluon(X_train, y_train, X_test, y_test, tag)
            print(f"  AutoGluon:    acc={ag_result['accuracy']:.4f} macro_f1={ag_result['macro_f1']:.4f} "
                  f"(best={ag_result['best_model']}, n_models={ag_result['n_models_trained']})")

            all_results.append({
                "frac": frac,
                "seed": seed,
                "n_train": n_train,
                "min_per_class": int(class_counts.min()),
                "max_per_class": int(class_counts.max()),
                "linear_probe": lp_result,
                "autogluon": ag_result,
            })

            # incremental save
            with open(RESULTS_DIR / "raw_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print("\nDone. Results saved to results/raw_results.json")


if __name__ == "__main__":
    main()
