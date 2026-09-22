"""
Build a multi-label manifest for the 1k-5k-label follow-up experiment.

Unlike scripts/01_build_manifest.py (which discarded every clip with more
than one species label, forcing a single-label multiclass framing), this
keeps ALL clips and treats WABAD as what it actually is: multi-label. Each
clip's target is a binary vector over the selected TARGET_SPECIES (present/
absent), using every clip in the pool as a positive example for the species
it contains and a negative example for the species it doesn't -- which is
what lets per-species positive counts reach 1k-5k+ instead of being capped
at a few hundred single-label clips.

Target species were chosen as the 8 with >=2000 total occurrences (any
co-occurrence, not just single-label) across all 72 WABAD sites, using the
sites where they occur most, so the audio download stays manageable
(~20 sites total, ~10GB) rather than needing all 72 sites (~20.6GB).
"""
import ast
import pandas as pd
from pathlib import Path

SITES = [
    "ARD", "BAM", "BERB", "HAG", "KIB", "SAL", "BIAL", "BMT",  # original 8
    "BOLIN", "CAT", "CLH", "DONG", "DYOM", "EFFOR", "EFFOU", "EVROS",
    "KAR", "NAV", "OLIV", "PITI", "POZO", "SCHF", "SITH", "SLOB", "VIL",
]

TARGET_SPECIES = [
    "Fringilla coelebs",
    "Malacopteron magnirostre",
    "Turdus merula",
    "Stachyris maculata",
    "Sylvia atricapilla",
    "Dicrurus paradiseus",
    "Erithacus rubecula",
    "Luscinia megarhynchos",
]

DATA_DIR = Path("/Users/SML161/automl/data")
OUT_PATH = Path("/Users/SML161/automl/data/manifest_multilabel.parquet")


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

    # Binary label columns: 1 if the species is present anywhere in the clip.
    for sp in TARGET_SPECIES:
        col = "label__" + sp.replace(" ", "_")
        df[col] = df["species_list"].apply(lambda x: sp in x).astype(int)

    label_cols = ["label__" + sp.replace(" ", "_") for sp in TARGET_SPECIES]

    # Keep clips that are relevant to at least one target species OR serve as
    # a negative example pool. Using ALL clips keeps negatives realistic
    # (background soundscape without the target species) rather than
    # sampling negatives only from clips that mention some OTHER rare
    # species, which would bias the negative class toward "has some bird".
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
    print(f"Wrote multi-label manifest with {len(manifest)} clips to {OUT_PATH}")
    print(manifest["split"].value_counts())
    print()
    print("Positive counts per species (train / test):")
    for sp, col in zip(TARGET_SPECIES, label_cols):
        tr_pos = manifest.loc[manifest["split"] == "train", col].sum()
        te_pos = manifest.loc[manifest["split"] == "test", col].sum()
        print(f"  {sp:30s} train={tr_pos:5d}  test={te_pos:5d}")


if __name__ == "__main__":
    main()
