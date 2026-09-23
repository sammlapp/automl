"""
Reuse the already-extracted Perch V2 embeddings (from
scripts/06_extract_embeddings_multilabel.py) for the all-species experiment,
instead of re-running ONNX inference on the same 47,701 clips.

Verified in scripts/09_build_all_species_manifest.py's manifest that the
clip order is identical to manifest_multilabel.parquet (same 25 sites, same
audio, same filename/site/split order) -- so we just pair the existing
embeddings array with the new 455-species label matrix.
"""
import numpy as np
import pandas as pd
from pathlib import Path

EXISTING_EMB_PATH = Path("/Users/SML161/automl/data/embeddings_multilabel.npz")
MANIFEST_PATH = Path("/Users/SML161/automl/data/manifest_all_species.parquet")
OUT_PATH = Path("/Users/SML161/automl/data/embeddings_all_species.npz")


def main():
    existing = np.load(EXISTING_EMB_PATH, allow_pickle=True)
    manifest = pd.read_parquet(MANIFEST_PATH)

    # Sanity check: same clip order as the manifest used for the existing embeddings.
    assert len(existing["filename"]) == len(manifest), "length mismatch"
    assert (existing["filename"] == manifest["filename"].values).all(), "filename order mismatch"
    assert (existing["site_ID"] == manifest["site_ID"].values).all(), "site order mismatch"
    assert (existing["split"] == manifest["split"].values).all(), "split order mismatch"

    label_cols = [c for c in manifest.columns if c.startswith("label__")]
    print(f"{len(label_cols)} species label columns")

    labels = manifest[label_cols].values.astype(np.int8)

    np.savez_compressed(
        OUT_PATH,
        embeddings=existing["embeddings"],
        valid_mask=existing["valid_mask"],
        labels=labels,
        label_names=np.array(label_cols),
        split=existing["split"],
        site_ID=existing["site_ID"],
        filename=existing["filename"],
    )
    print(f"Saved all-species embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
