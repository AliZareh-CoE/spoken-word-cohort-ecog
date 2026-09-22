"""Measures the onset of every speech sound from the audio, word by word, for the words whose onsets had been interpolated.

Writes results/features/phonemes_realigned.parquet.
"""
from __future__ import annotations

import os

import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
MAIN = Path(os.environ.get("PODCAST_ROOT", Path(__file__).resolve().parent))
CHARSIU_COMMIT = "13a69f2a22ca0c0962b75cc693399b0ae23a12c9"
sys.path.insert(0, str(HERE / "vendor" / "charsiu" / "src"))
sys.path.insert(0, str(MAIN))

PAD, PRE_PAD, TOL, FS = 0.10, 0.0, 0.05, 16000


def merge_repeats(phones):
    out = []
    for t0, t1, lab in phones:
        if lab == "[SIL]":
            continue
        if out and out[-1][2] == lab:
            out[-1] = (out[-1][0], t1, lab)
        else:
            out.append((t0, t1, lab))
    return out


def accept(aligned, ours, w_on, w_off):
    """Pre-declared acceptance: identical label sequence, onsets within tolerance, strictly increasing."""
    if [lab for _, _, lab in aligned] != ours:
        return False
    on = [t0 for t0, _, _ in aligned]
    return all(w_on - TOL <= t <= w_off + TOL for t in on) and all(b > a for a, b in zip(on, on[1:]))


def main():
    from Charsiu import charsiu_forced_aligner
    words = pd.read_parquet(MAIN / "data" / "features" / "words.parquet").set_index("word_idx")
    phon = pd.read_parquet(MAIN / "data" / "features" / "phonemes_hybrid.parquet")
    audio, sr = sf.read(str(MAIN / "data" / "ds005574" / "stimuli" / "podcast.wav"), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    a16 = librosa.resample(audio, orig_sr=sr, target_sr=FS)
    al = charsiu_forced_aligner(aligner="charsiu/en_w2v2_fc_10ms")
    out = phon.copy()
    report = []
    target = phon[phon.phoneme_alignment_method != "charsiu"].word_idx.unique()
    for n, wi in enumerate(target):
        w = words.loc[wi]
        rows = phon[phon.word_idx == wi].sort_values("phoneme_idx")
        ours = rows.phoneme_arpabet.str.replace(r"\d", "", regex=True).tolist()
        t0 = max(0.0, float(w.onset_sec) - PRE_PAD)
        chunk = a16[int(t0 * FS):int((float(w.offset_sec) + PAD) * FS)]
        status = "rejected"
        try:
            aligned = [(t0 + a, t0 + b, lab) for a, b, lab in merge_repeats(al.align(chunk, str(w.word))[0])]
            if accept(aligned, ours, float(w.onset_sec), float(w.offset_sec)):
                status = "accepted"
                for (on, off, _), idx in zip(aligned, rows.index):
                    out.loc[idx, "phoneme_onset_sec"] = on
                    out.loc[idx, "phoneme_offset_sec"] = off
                    out.loc[idx, "phoneme_duration_sec"] = off - on
                    out.loc[idx, "phoneme_alignment_method"] = "charsiu-word"
        except Exception as e:
            status = f"error: {type(e).__name__}"
        report.append(dict(word_idx=int(wi), word=str(w.word), n_phonemes=len(ours), status=status))
        if n % 200 == 0:
            print(n, len(target), flush=True)
    rep = pd.DataFrame(report)
    (HERE / "results" / "features").mkdir(parents=True, exist_ok=True)
    out.to_parquet(HERE / "results" / "features" / "phonemes_realigned.parquet", index=False)
    rep.to_csv(HERE / "results" / "features" / "alignment_report.csv", index=False)
    acc = rep.status.eq("accepted")
    print(f"accepted words {acc.sum()}/{len(rep)} ({acc.mean():.1%}); phonemes {rep.n_phonemes[acc].sum()}/{rep.n_phonemes.sum()}")
    print(out.phoneme_alignment_method.value_counts())


if __name__ == "__main__":
    main()
