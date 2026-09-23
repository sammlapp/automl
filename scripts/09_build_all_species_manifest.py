"""
Build a multi-label manifest covering ALL species with enough data to
support the small-training-size experiment (targets: 5, 25, 100 positive
training examples per species -- see scripts/10_run_all_species_experiment.py).

Reuses the same 25 already-downloaded WABAD sites and the same audio/
embeddings as the 8-species Phase 2 manifest (scripts/05_build_multilabel_manifest.py)
-- no new downloads needed. Only the label columns differ: instead of 8
hand-picked species, every species with enough occurrences to support at
least the smallest target (>=5 train positives, >=1 test positive) gets a
binary presence/absence column.
"""
import ast
import pandas as pd
from pathlib import Path

SITES = [
    "ARD", "BAM", "BERB", "HAG", "KIB", "SAL", "BIAL", "BMT",
    "BOLIN", "CAT", "CLH", "DONG", "DYOM", "EFFOR", "EFFOU", "EVROS",
    "KAR", "NAV", "OLIV", "PITI", "POZO", "SCHF", "SITH", "SLOB", "VIL",
]

MIN_TRAIN_POS = 5
MIN_TEST_POS = 1

DATA_DIR = Path("/Users/SML161/automl/data")
OUT_PATH = Path("/Users/SML161/automl/data/manifest_all_species.parquet")
SPECIES_LIST_PATH = Path("/Users/SML161/automl/data/all_species_list.csv")


def resolve_path(row, audio_root):
    candidates = [
        audio_root / row["site_ID"] / "audio" / row["filename"],
        audio_root / row["site_ID"] / row["filename"],
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def main():
    frames = []
    for s in SITES:
        tr = pd.read_parquet(DATA_DIR / s / f"{s}_metadata_train.parquet")
        te = pd.read_parquet(DATA_DIR / s / f"{s}_metadata_test.parquet")
        tr["split"] = "train"
        te["split"] = "test"
        frames.append(pd.concat([tr, te], ignore_index=True))
    df = pd.concat(frames, ignore_index=True)
    print(f"Total clips across {len(SITES)} sites: {len(df)}")

    df["species_list"] = df["species"].apply(ast.literal_eval)

    from collections import defaultdict
    train_pos = defaultdict(int)
    test_pos = defaultdict(int)
    for _, row in df.iterrows():
        for sp in set(row["species_list"]):
            if row["split"] == "train":
                train_pos[sp] += 1
            else:
                test_pos[sp] += 1

    target_species = sorted(
        sp for sp in train_pos
        if train_pos[sp] >= MIN_TRAIN_POS and test_pos.get(sp, 0) >= MIN_TEST_POS
    )
    print(f"Species with >={MIN_TRAIN_POS} train positives and >={MIN_TEST_POS} test positives: {len(target_species)}")

    species_df = pd.DataFrame({
        "species": target_species,
        "train_pos": [train_pos[sp] for sp in target_species],
        "test_pos": [test_pos.get(sp, 0) for sp in target_species],
    })
    species_df.to_csv(SPECIES_LIST_PATH, index=False)
    print(f"Wrote species list to {SPECIES_LIST_PATH}")

    for sp in target_species:
        col = "label__" + sp.replace(" ", "_")
        df[col] = df["species_list"].apply(lambda x: sp in x).astype(int)

    label_cols = ["label__" + sp.replace(" ", "_") for sp in target_species]

    manifest = df[
        ["filename", "filepath", "site_ID", "split", "recording_location", "sampling_rate"]
        + label_cols
    ].copy()

    audio_root = DATA_DIR.parent / "data_audio"
    manifest["local_path"] = manifest.apply(lambda r: resolve_path(r, audio_root), axis=1)
    missing = (~manifest["local_path"].apply(lambda p: Path(p).exists())).sum()
    if missing:
        print(f"WARNING: {missing} manifest clips have no resolvable local audio file")

    manifest.to_parquet(OUT_PATH, index=False)
    print(f"Wrote all-species multi-label manifest with {len(manifest)} clips, "
          f"{len(label_cols)} species columns, to {OUT_PATH}")


if __name__ == "__main__":
    main()
