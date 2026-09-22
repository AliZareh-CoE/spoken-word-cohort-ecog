"""How far the measured onset of every speech sound lies outside the interval of its word in the transcript, by alignment
method.

Writes results/alignment_quality.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

ph = pd.read_parquet(common.HERE / "results" / "features" / "phonemes_realigned.parquet")
w = pd.read_parquet(common.MAIN / "data" / "features" / "words.parquet", columns=["word_idx", "onset_sec", "offset_sec"])
x = ph.merge(w, on="word_idx", how="left")
outside = np.maximum(np.maximum(x.onset_sec - x.phoneme_onset_sec, x.phoneme_onset_sec - x.offset_sec), 0.0)
x = x.assign(outside=outside)
rows = []
for m, g in x.groupby("phoneme_alignment_method"):
    far = g[g.outside > 1.0]
    rows.append(dict(method=m, n_phonemes=len(g), share_of_all=len(g) / len(x), share_gt_50ms_outside_word=float((g.outside > 0.05).mean()), share_gt_200ms=float((g.outside > 0.2).mean()),
                     n_gt_1s=int(len(far)), n_words_gt_1s=int(far.word_idx.nunique()), max_s=float(g.outside.max())))
out = pd.DataFrame(rows)
f = common.OUT / "alignment_quality.csv"
if f.exists():
    old = pd.read_csv(f)
    assert list(old.method) == list(out.method) and np.allclose(old.drop(columns="method").to_numpy(float), out.drop(columns="method").to_numpy(float), atol=1e-9), "does not reproduce the recorded file"
out.to_csv(f, index=False)
print(out.to_string(index=False))
