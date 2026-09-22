"""
Extract Perch V2 embeddings for every clip in the manifest.

Perch V2 (headless ONNX) expects raw waveform input of shape [batch, 160000]
(5 seconds @ 32kHz). WABAD clips are longer field recordings at various native
sample rates; we resample to 32kHz, then average embeddings over non-overlapping
5-second windows (center-cropping/padding the last partial window) to get one
fixed-length embedding per clip, matching how Perch-style models are commonly
applied to longer soundscape clips for transfer-learning benchmarks.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

MANIFEST_PATH = Path("/Users/SML161/automl/data/manifest.parquet")
MODEL_PATH = Path("/Users/SML161/automl/models/perch_v2_embedding_only.onnx")
OUT_PATH = Path("/Users/SML161/automl/data/embeddings.npz")

TARGET_SR = 32000
WINDOW_SAMPLES = 160000  # 5s @ 32kHz


def load_audio_windows(path, target_sr=TARGET_SR, window_samples=WINDOW_SAMPLES):
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    n = len(wav)
    if n == 0:
        return None
    if n < window_samples:
        wav = np.pad(wav, (0, window_samples - n))
        n = window_samples
    n_windows = max(1, n // window_samples)
    windows = []
    for i in range(n_windows):
        seg = wav[i * window_samples : (i + 1) * window_samples]
        if len(seg) < window_samples:
            seg = np.pad(seg, (0, window_samples - len(seg)))
        windows.append(seg)
    return np.stack(windows).astype(np.float32)


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    sess = ort.InferenceSession(str(MODEL_PATH))
    input_name = sess.get_inputs()[0].name

    embeddings = np.zeros((len(manifest), 1536), dtype=np.float32)
    valid_mask = np.zeros(len(manifest), dtype=bool)

    for i, row in tqdm(manifest.iterrows(), total=len(manifest)):
        windows = load_audio_windows(row["local_path"])
        if windows is None:
            continue
        try:
            out = sess.run(None, {input_name: windows})[0]  # [n_windows, 1536]
        except Exception as e:
            print(f"Inference failed for {row['local_path']}: {e}")
            continue
        embeddings[i] = out.mean(axis=0)
        valid_mask[i] = True

    print(f"Successfully embedded {valid_mask.sum()} / {len(manifest)} clips")

    np.savez_compressed(
        OUT_PATH,
        embeddings=embeddings,
        valid_mask=valid_mask,
        species_name=manifest["species_name"].values,
        split=manifest["split"].values,
        site_ID=manifest["site_ID"].values,
        filename=manifest["filename"].values,
    )
    print(f"Saved embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
