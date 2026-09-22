"""
Build a single-label multi-class manifest from WABAD metadata across selected sites.

Selects species with >= MIN_CLIPS single-label clips (clips annotated with exactly
one species), pooling across sites. Keeps the dataset's native train/test split
assignment (per-site) as the base held-out eval set, and additionally will be
resampled at various training fractions downstream.
"""
import ast
import pandas as pd
from pathlib import Path

SITES = ["ARD", "BAM", "BERB", "HAG", "KIB", "SAL", "BIAL", "BMT"]
MIN_CLIPS = 40
DATA_DIR = Path("/Users/SML161/automl/data")
OUT_PATH = Path("/Users/SML161/automl/data/manifest.parquet")


def main():
    frames = []
    for s in SITES:
        tr = pd.read_parquet(DATA_DIR / s / f"{s}_metadata_train.parquet")
        te = pd.read_parquet(DATA_DIR / s / f"{s}_metadata_test.parquet")
        tr["split"] = "train"
        te["split"] = "test"
        frames.append(pd.concat([tr, te], ignore_index=True))
    df = pd.concat(frames, ignore_index=True)

    df["species_list"] = df["species"].apply(ast.literal_eval)
    df["n_species"] = df["species_list"].apply(len)
    single = df[df["n_species"] == 1].copy()
    single["species_name"] = single["species_list"].apply(lambda x: x[0])

    counts = single.groupby("species_name").size().sort_values(ascending=False)
    keep_species = counts[counts >= MIN_CLIPS].index.tolist()
    print(f"Selected {len(keep_species)} species with >= {MIN_CLIPS} single-label clips")
    print(counts[counts >= MIN_CLIPS])

    manifest = single[single["species_name"].isin(keep_species)].copy()
    manifest = manifest[
        [
            "filename",
            "filepath",
            "species_name",
            "site_ID",
            "split",
            "recording_location",
            "sampling_rate",
        ]
    ].reset_index(drop=True)

    # local audio path within extracted site zip. Zip layout is inconsistent across
    # sites: some extract to data_audio/<site>/audio/<filename>, others flat to
    # data_audio/<site>/<filename>. Resolve whichever actually exists on disk.
    audio_root = DATA_DIR.parent / "data_audio"

    def resolve_path(row):
        candidates = [
            audio_root / row["site_ID"] / "audio" / row["filename"],
            audio_root / row["site_ID"] / row["filename"],
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(candidates[0])  # fall back; extraction step will report as missing

    manifest["local_path"] = manifest.apply(resolve_path, axis=1)
    missing = (~manifest["local_path"].apply(lambda p: Path(p).exists())).sum()
    if missing:
        print(f"WARNING: {missing} manifest clips have no resolvable local audio file")

    manifest.to_parquet(OUT_PATH, index=False)
    print(f"Wrote manifest with {len(manifest)} clips to {OUT_PATH}")
    print(manifest["split"].value_counts())
    print(manifest.groupby(["species_name", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
