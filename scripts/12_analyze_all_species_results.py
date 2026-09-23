"""
Aggregate results/raw_results_all_species.json (455 species, small training
targets 5/25/100 positives/species, single seed) into:
  - mAP / macro-AUROC per target_pos (headline numbers, mean across all
    species that were run at that target)
  - per-species table for reference
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("/Users/SML161/automl/results")


def main():
    with open(RESULTS_DIR / "raw_results_all_species.json") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        for model_key in ("linear_probe", "autogluon"):
            m = r[model_key]
            rows.append({
                "species": r["species"],
                "target_pos": r["target_pos"],
                "n_train": r["n_train"],
                "n_train_pos": r["n_train_pos"],
                "n_test_pos": r["n_test_pos"],
                "n_test": r["n_test"],
                "model": model_key,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "average_precision": m["average_precision"],
                "auroc": m["auroc"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "all_species_summary_long.csv", index=False)

    print(f"Total species-target-model rows: {len(df)}")
    print(f"Distinct species: {df['species'].nunique()}")
    for t in sorted(df["target_pos"].unique()):
        n_sp = df[(df["target_pos"] == t) & (df["model"] == "linear_probe")]["species"].nunique()
        print(f"  target_pos={t}: {n_sp} species")

    # headline: mAP and macro-AUROC per target_pos, per model
    headline = df.groupby(["target_pos", "model"]).agg(
        mAP=("average_precision", "mean"),
        macro_AUROC=("auroc", "mean"),
        n_species=("species", "nunique"),
    ).reset_index()
    headline.to_csv(RESULTS_DIR / "all_species_headline.csv", index=False)

    print("\n=== Headline: mAP and macro-AUROC by training-positives target (ALL SPECIES) ===")
    print(headline.to_string(index=False))

    pivot_map = headline.pivot(index="target_pos", columns="model", values="mAP")
    pivot_map["delta_mAP_ag_minus_lp"] = pivot_map["autogluon"] - pivot_map["linear_probe"]
    pivot_auroc = headline.pivot(index="target_pos", columns="model", values="macro_AUROC")
    pivot_auroc["delta_macroAUROC_ag_minus_lp"] = pivot_auroc["autogluon"] - pivot_auroc["linear_probe"]

    print("\n=== mAP delta (AutoGluon - Linear Probe) ===")
    print(pivot_map.to_string())
    print("\n=== macro-AUROC delta (AutoGluon - Linear Probe) ===")
    print(pivot_auroc.to_string())

    pivot_map.to_csv(RESULTS_DIR / "all_species_delta_mAP.csv")
    pivot_auroc.to_csv(RESULTS_DIR / "all_species_delta_macroAUROC.csv")

    # win/tie/loss counts per target
    print("\n=== Win/loss counts (AP) per target ===")
    for t in sorted(df["target_pos"].unique()):
        sub = df[df["target_pos"] == t]
        pivot = sub.pivot(index="species", columns="model", values="average_precision")
        wins_ag = (pivot["autogluon"] > pivot["linear_probe"]).sum()
        wins_lp = (pivot["autogluon"] < pivot["linear_probe"]).sum()
        ties = (pivot["autogluon"] == pivot["linear_probe"]).sum()
        print(f"  target_pos={t}: AutoGluon wins {wins_ag}, Linear probe wins {wins_lp}, ties {ties} (n={len(pivot)})")

    # per-species table at each target for reference
    for t in sorted(df["target_pos"].unique()):
        sub = df[df["target_pos"] == t]
        wide = sub.pivot(index="species", columns="model", values=["average_precision", "auroc"])
        wide.columns = ["_".join(c) for c in wide.columns]
        wide["delta_ap"] = wide["average_precision_autogluon"] - wide["average_precision_linear_probe"]
        wide = wide.sort_values("delta_ap", ascending=False)
        wide.to_csv(RESULTS_DIR / f"all_species_per_species_target{t}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
