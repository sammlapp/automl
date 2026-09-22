"""
Multi-label follow-up: linear probe vs AutoGluon on species with 1k-5k
training labels, framed as one-vs-rest binary presence/absence classification
per species (the natural framing for WABAD, which is inherently multi-label --
see scripts/05_build_multilabel_manifest.py for why).

For each target species and each training-size target in TRAIN_SIZE_TARGETS,
we subsample the (large) native training pool down to approximately that many
POSITIVE examples (keeping a matched, class-balanced sample of negatives --
see note below), fit both models, and evaluate on the FULL native test split
(all clips, not subsampled).

Primary eval metrics (per scripts/08_analyze_multilabel_results.py, which
aggregates across species): mAP (mean average precision, averaged across
species) and macro-AUROC (AUROC averaged across species), plus per-class AP
and per-class AUROC. AP and AUROC are threshold-free ranking metrics, the
standard choice for imbalanced binary/multi-label bioacoustic detection.
Accuracy/F1/precision/recall are also recorded per-class for reference but
are not the headline metrics (they require picking a decision threshold).
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

EMB_PATH = Path("/Users/SML161/automl/data/embeddings_multilabel.npz")
RESULTS_DIR = Path("/Users/SML161/automl/results")
AG_MODEL_DIR = Path("/Users/SML161/automl/models/autogluon_multilabel_tmp")

# Approximate number of POSITIVE training examples to subsample to. Negatives
# are matched at a fixed 3:1 negative:positive ratio (rather than using all
# available negatives, which would be tens of thousands and make the two
# classifiers' training set sizes hard to compare across species/fractions).
TRAIN_POS_TARGETS = [1000, 2500, 5000]
NEG_POS_RATIO = 3
SEEDS = [0, 1, 2]


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
    # roc_auc_score requires both classes present in y_true; guard for the
    # (unlikely, given the test split is fixed and never subsampled) case
    # where a species has zero test-set positives or negatives.
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

    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
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

    # Same stable model-family restriction as scripts/03_run_experiment.py:
    # GBM/CAT/XGB/FASTAI are unusable on this machine (segfaults / pathological
    # slowness inside AutoGluon's fit wrapper -- see PROGRESS.md). Using
    # RF + ExtraTrees + PyTorch NN + weighted ensemble.
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
        time_limit=120,
    )

    preds = predictor.predict(test_df).values.astype(int)
    scores = predictor.predict_proba(test_df)[1].values
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]["model"] if len(leaderboard) else None
    n_models = len(leaderboard)

    shutil.rmtree(save_path, ignore_errors=True)

    metrics = binary_metrics(y_test, preds, scores)
    metrics["best_model"] = best_model
    metrics["n_models_trained"] = n_models
    return metrics


def main():
    X, labels, label_names, split = load_data()
    train_mask = split == "train"
    test_mask = split == "test"

    print(f"Total clips: {len(X)} (train={train_mask.sum()}, test={test_mask.sum()})")
    print(f"Species: {label_names}")

    AG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "raw_results_multilabel.json"
    all_results = []
    done_tags = set()
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        done_tags = {f"{r['species']}_pos{r['target_pos']}_seed{r['seed']}" for r in all_results}
        print(f"Resuming: {len(done_tags)} combos already done")

    for sp_idx, species in enumerate(label_names):
        y_all = labels[:, sp_idx]
        y_train_all = y_all[train_mask]
        y_test = y_all[test_mask]
        X_test = X[test_mask]

        pos_idx_all = np.where(train_mask)[0][y_train_all == 1]
        neg_idx_all = np.where(train_mask)[0][y_train_all == 0]

        n_test_pos = int(y_test.sum())
        print(f"\n### Species: {species} | train pool pos={len(pos_idx_all)} neg={len(neg_idx_all)} "
              f"| test pos={n_test_pos}/{len(y_test)} ###")

        if len(pos_idx_all) < 50:
            print(f"  Skipping {species}: fewer than 50 positive training examples available")
            continue

        for target_pos in TRAIN_POS_TARGETS:
            for seed in SEEDS:
                if target_pos >= len(pos_idx_all) and seed != SEEDS[0]:
                    continue  # deterministic (uses all positives) -- skip repeat seeds

                tag = f"{species}_pos{target_pos}_seed{seed}"
                if tag in done_tags:
                    print(f"  Skipping {tag}: already done")
                    continue

                sub_idx, n_pos, n_neg = subsample_train(pos_idx_all, neg_idx_all, target_pos, NEG_POS_RATIO, seed)
                X_train, y_train = X[sub_idx], y_all[sub_idx]

                print(f"\n  === {tag}: n_train={len(sub_idx)} (pos={n_pos}, neg={n_neg}) ===")

                lp_result = fit_eval_linear_probe(X_train, y_train, X_test, y_test)
                print(f"    Linear probe: acc={lp_result['accuracy']:.4f} f1={lp_result['f1']:.4f} "
                      f"AP={lp_result['average_precision']:.4f}")

                ag_result = fit_eval_autogluon(X_train, y_train, X_test, y_test, tag.replace(" ", "_"))
                print(f"    AutoGluon:    acc={ag_result['accuracy']:.4f} f1={ag_result['f1']:.4f} "
                      f"AP={ag_result['average_precision']:.4f} (best={ag_result['best_model']})")

                all_results.append({
                    "species": species,
                    "target_pos": target_pos,
                    "seed": seed,
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

    print("\nDone. Results saved to results/raw_results_multilabel.json")


if __name__ == "__main__":
    main()
