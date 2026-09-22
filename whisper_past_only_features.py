"""Whisper encoder features that use past audio only: for each 20-ms frame the encoder receives the 30 s of audio ending at
that frame, and the last encoder frame is kept. A control variant also receives the next 500 ms.

Writes results/features/whisper_causal.hdf5 and whisper_causal_la500.hdf5.
"""
from __future__ import annotations

import argparse
import subprocess
import time

import h5py
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

import common

MODEL_ID = "openai/whisper-medium.en"
SR, HOP, WIN, N_FRAMES, LA = 16000, 320, 480000, 90000, 25
OUTDIR = common.OUT / "features"
PARTS = OUTDIR / "whisper_causal_parts"
PART = 3000


def load_audio():
    wav = common.MAIN / "data" / "ds005574" / "stimuli" / "podcast.wav"
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(wav), "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float32) / 32768.0


def window(padded: np.ndarray, k: int) -> np.ndarray:
    """30 s of audio ending at the end of frame k. `padded` = WIN zeros + audio."""
    end = (k + 1) * HOP + WIN
    return padded[end - WIN:end]


def pad_audio(audio: np.ndarray) -> np.ndarray:
    need = N_FRAMES * HOP
    a = audio[:need] if audio.size >= need else np.concatenate([audio, np.zeros(need - audio.size, np.float32)])
    return np.concatenate([np.zeros(WIN, np.float32), a])


class Extractor:
    def __init__(self, dtype: str = "fp16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
        self.enc = WhisperModel.from_pretrained(MODEL_ID).encoder.eval().to(self.device)
        self.dtype = torch.float16 if (dtype == "fp16" and self.device == "cuda") else torch.float32
        if self.dtype == torch.float16:
            self.enc = self.enc.half()

    @torch.no_grad()
    def __call__(self, windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        feats = self.fe(windows, sampling_rate=SR, return_tensors="pt", device=self.device)["input_features"]
        h = self.enc(feats.to(self.device, self.dtype)).last_hidden_state
        return h[:, -1].float().cpu().numpy(), h[:, -1 - LA].float().cpu().numpy()


def run(batch: int, dtype: str):
    PARTS.mkdir(parents=True, exist_ok=True)
    padded = pad_audio(load_audio())
    ex = Extractor(dtype)
    t0 = time.time()
    for p0 in range(0, N_FRAMES, PART):
        f = PARTS / f"part_{p0:06d}.npz"
        if f.exists():
            continue
        wc, la = [], []
        for b0 in range(p0, min(p0 + PART, N_FRAMES), batch):
            ks = range(b0, min(b0 + batch, p0 + PART, N_FRAMES))
            a, b = ex([window(padded, k) for k in ks])
            wc.append(a)
            la.append(b)
        np.savez(f, wc=np.concatenate(wc).astype(np.float32), la=np.concatenate(la).astype(np.float32))
        print(f"[A28] frames {p0}-{p0 + PART} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    wc = np.concatenate([np.load(PARTS / f"part_{p0:06d}.npz")["wc"] for p0 in range(0, N_FRAMES, PART)])
    la = np.concatenate([np.load(PARTS / f"part_{p0:06d}.npz")["la"] for p0 in range(0, N_FRAMES, PART)])
    assert wc.shape == la.shape == (N_FRAMES, 1024)
    wc500 = np.empty_like(la)
    wc500[: N_FRAMES - LA] = la[LA:]
    with torch.no_grad():
        feats = ex.fe([window(padded, N_FRAMES - 1)], sampling_rate=SR, return_tensors="pt", device=ex.device)["input_features"]
        h = ex.enc(feats.to(ex.device, ex.dtype)).last_hidden_state[0].float().cpu().numpy()
    wc500[N_FRAMES - LA:] = h[1500 - LA:]
    for name, arr in (("whisper_causal.hdf5", wc), ("whisper_causal_la500.hdf5", wc500)):
        with h5py.File(OUTDIR / name, "w") as f:
            f.create_dataset("vectors", data=arr.T.astype(np.float32))
    print("[A28] wrote", OUTDIR / "whisper_causal.hdf5", "and whisper_causal_la500.hdf5", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", default="fp16")
    a = ap.parse_args()
    if a.time:
        padded = pad_audio(load_audio())
        ex = Extractor(a.dtype)
        ex([window(padded, 50000 + i) for i in range(a.batch)])
        torch.cuda.synchronize()
        t0 = time.time()
        for b0 in range(0, a.time, a.batch):
            ex([window(padded, 50000 + b0 + i) for i in range(a.batch)])
        torch.cuda.synchronize()
        dt = (time.time() - t0) / a.time
        print(f"[A28] {a.dtype} batch {a.batch}: {dt * 1e3:.1f} ms/pass -> projected {dt * N_FRAMES / 3600:.2f} h for {N_FRAMES} passes")
    if a.run:
        run(a.batch, a.dtype)
