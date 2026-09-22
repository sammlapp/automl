"""
Summarize raw_results.json into tables and a plot comparing linear probe vs
AutoGluon accuracy/macro-F1 across training fractions, plus per-class deltas
at the largest training fraction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("/Users/SML161/automl/results")


def main():
    with open(RESULTS_DIR / "raw_results.json") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        rows.append({
            "frac": r["frac"],
            "seed": r["seed"],
            "n_train": r["n_train"],
            "model": "linear_probe",
            "accuracy": r["linear_probe"]["accuracy"],
            "macro_f1": r["linear_probe"]["macro_f1"],
        })
        rows.append({
            "frac": r["frac"],
            "seed": r["seed"],
            "n_train": r["n_train"],
            "model": "autogluon",
            "accuracy": r["autogluon"]["accuracy"],
            "macro_f1": r["autogluon"]["macro_f1"],
            "best_model": r["autogluon"].get("best_model"),
            "n_models_trained": r["autogluon"].get("n_models_trained"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary_long.csv", index=False)

    # aggregate across seeds
    agg = df.groupby(["frac", "model"]).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_n_train=("n_train", "mean"),
    ).reset_index()
    print("=== Aggregated results (mean over seeds) ===")
    print(agg.to_string(index=False))
    agg.to_csv(RESULTS_DIR / "summary_aggregated.csv", index=False)

    # pivot for easy delta computation
    pivot_f1 = agg.pivot(index="frac", columns="model", values="mean_macro_f1")
    pivot_f1["delta_macro_f1_ag_minus_lp"] = pivot_f1["autogluon"] - pivot_f1["linear_probe"]
    pivot_acc = agg.pivot(index="frac", columns="model", values="mean_accuracy")
    pivot_acc["delta_accuracy_ag_minus_lp"] = pivot_acc["autogluon"] - pivot_acc["linear_probe"]

    print("\n=== Macro-F1 delta (AutoGluon - Linear Probe) ===")
    print(pivot_f1.to_string())
    print("\n=== Accuracy delta (AutoGluon - Linear Probe) ===")
    print(pivot_acc.to_string())

    pivot_f1.to_csv(RESULTS_DIR / "delta_macro_f1.csv")
    pivot_acc.to_csv(RESULTS_DIR / "delta_accuracy.csv")

    # per-class F1 comparison at the largest fraction
    largest_frac = max(r["frac"] for r in raw)
    per_class_rows = []
    for r in raw:
        if r["frac"] != largest_frac:
            continue
        lp_pc = r["linear_probe"]["per_class_f1"]
        ag_pc = r["autogluon"]["per_class_f1"]
        for cls in lp_pc:
            per_class_rows.append({
                "seed": r["seed"],
                "species": cls,
                "linear_probe_f1": lp_pc[cls],
                "autogluon_f1": ag_pc.get(cls, np.nan),
            })
    pc_df = pd.DataFrame(per_class_rows)
    pc_agg = pc_df.groupby("species").agg(
        linear_probe_f1=("linear_probe_f1", "mean"),
        autogluon_f1=("autogluon_f1", "mean"),
    ).reset_index()
    pc_agg["delta"] = pc_agg["autogluon_f1"] - pc_agg["linear_probe_f1"]
    pc_agg = pc_agg.sort_values("delta", ascending=False)
    pc_agg.to_csv(RESULTS_DIR / f"per_class_f1_frac{largest_frac}.csv", index=False)
    print(f"\n=== Per-class F1 at frac={largest_frac} (sorted by AutoGluon advantage) ===")
    print(pc_agg.to_string(index=False))

    print(f"\nAutoGluon wins on {(pc_agg['delta']>0).sum()}/{len(pc_agg)} classes at frac={largest_frac}")
    print(f"Linear probe wins on {(pc_agg['delta']<0).sum()}/{len(pc_agg)} classes")
    print(f"Ties on {(pc_agg['delta']==0).sum()}/{len(pc_agg)} classes")


if __name__ == "__main__":
    main()
