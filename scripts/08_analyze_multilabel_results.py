"""
Aggregate results/raw_results_multilabel.json into:
  - per-species, per-train-size-target AP and AUROC (both models)
  - mAP (mean AP across species) and macro-AUROC (mean AUROC across species),
    per train-size target -- the headline comparison
  - per-class AP/AUROC tables at each train-size target
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("/Users/SML161/automl/results")


def main():
    with open(RESULTS_DIR / "raw_results_multilabel.json") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        for model_key, model_name in [("linear_probe", "linear_probe"), ("autogluon", "autogluon")]:
            m = r[model_key]
            rows.append({
                "species": r["species"],
                "target_pos": r["target_pos"],
                "seed": r["seed"],
                "n_train": r["n_train"],
                "n_train_pos": r["n_train_pos"],
                "n_test_pos": r["n_test_pos"],
                "n_test": r["n_test"],
                "model": model_name,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "average_precision": m["average_precision"],
                "auroc": m["auroc"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "multilabel_summary_long.csv", index=False)

    # per-species, per-target_pos, per-model: mean over seeds
    per_species = df.groupby(["species", "target_pos", "model"]).agg(
        mean_ap=("average_precision", "mean"),
        std_ap=("average_precision", "std"),
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_f1=("f1", "mean"),
        mean_n_train=("n_train", "mean"),
        mean_n_train_pos=("n_train_pos", "mean"),
    ).reset_index()
    per_species.to_csv(RESULTS_DIR / "multilabel_per_species.csv", index=False)

    print("=== Per-species AP / AUROC (mean over seeds) ===")
    print(per_species.to_string(index=False))

    # headline: mAP and macro-AUROC per target_pos, per model (mean across species)
    headline = per_species.groupby(["target_pos", "model"]).agg(
        mAP=("mean_ap", "mean"),
        macro_AUROC=("mean_auroc", "mean"),
    ).reset_index()
    headline.to_csv(RESULTS_DIR / "multilabel_headline.csv", index=False)

    print("\n=== Headline: mAP and macro-AUROC by training-positives target ===")
    print(headline.to_string(index=False))

    pivot_map = headline.pivot(index="target_pos", columns="model", values="mAP")
    pivot_map["delta_mAP_ag_minus_lp"] = pivot_map["autogluon"] - pivot_map["linear_probe"]
    pivot_auroc = headline.pivot(index="target_pos", columns="model", values="macro_AUROC")
    pivot_auroc["delta_macroAUROC_ag_minus_lp"] = pivot_auroc["autogluon"] - pivot_auroc["linear_probe"]

    print("\n=== mAP delta (AutoGluon - Linear Probe) ===")
    print(pivot_map.to_string())
    print("\n=== macro-AUROC delta (AutoGluon - Linear Probe) ===")
    print(pivot_auroc.to_string())

    pivot_map.to_csv(RESULTS_DIR / "multilabel_delta_mAP.csv")
    pivot_auroc.to_csv(RESULTS_DIR / "multilabel_delta_macroAUROC.csv")

    # per-class AP/AUROC at the largest training-positives target actually reached
    largest_target = int(per_species["target_pos"].max())
    pc = per_species[per_species["target_pos"] == largest_target].copy()
    pc_wide_ap = pc.pivot(index="species", columns="model", values="mean_ap")
    pc_wide_ap["delta_ap"] = pc_wide_ap["autogluon"] - pc_wide_ap["linear_probe"]
    pc_wide_auroc = pc.pivot(index="species", columns="model", values="mean_auroc")
    pc_wide_auroc["delta_auroc"] = pc_wide_auroc["autogluon"] - pc_wide_auroc["linear_probe"]

    pc_wide_ap = pc_wide_ap.sort_values("delta_ap", ascending=False)
    pc_wide_auroc = pc_wide_auroc.sort_values("delta_auroc", ascending=False)

    print(f"\n=== Per-class AP at target_pos={largest_target} ===")
    print(pc_wide_ap.to_string())
    print(f"\n=== Per-class AUROC at target_pos={largest_target} ===")
    print(pc_wide_auroc.to_string())

    pc_wide_ap.to_csv(RESULTS_DIR / f"multilabel_per_class_ap_target{largest_target}.csv")
    pc_wide_auroc.to_csv(RESULTS_DIR / f"multilabel_per_class_auroc_target{largest_target}.csv")

    print(f"\nAutoGluon wins on AP: {(pc_wide_ap['delta_ap']>0).sum()}/{len(pc_wide_ap)} species")
    print(f"AutoGluon wins on AUROC: {(pc_wide_auroc['delta_auroc']>0).sum()}/{len(pc_wide_auroc)} species")


if __name__ == "__main__":
    main()
