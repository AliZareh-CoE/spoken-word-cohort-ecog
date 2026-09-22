"""The cohort contribution with one deep model added to the spectrogram model: GPT-2 for the current or the previous word,
Whisper, and an untrained copy of Whisper; and the differences between them on a common electrode set.

Writes results/deep_models.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
PP = [f"matched{k}" for k in range(1, 6)]
TAG = "_pen7x60_ev50_onsets"
THR = 0.05
MODELS = ("spectrogram", "gpt2_current", "gpt2_previous", "whisper", "whisper_untrained")


def load(variant):
    fr = {}
    for c in ["real"] + PP:
        paths = [SH / f"{variant}_{c}{TAG}_{s}.parquet" for s in SUBS]
        if not all(p.exists() for p in paths):
            return None
        fr[c] = pd.concat([pd.read_parquet(p) for p in paths]).set_index(["subject", "electrode"]).cv_r
    return pd.DataFrame(fr)


def summarise(x, label, **extra):
    sm = x.groupby("subject").median()
    lo, hi = common.boot_ci(sm.values)
    return dict(label=label, n_elec=int(x.size), n_subj=int(sm.size), subj_median=float(sm.median()), ci_lo=lo, ci_hi=hi,
                n_subj_pos=int((sm > 0).sum()), p_two_sided=common.wilcoxon_two_sided(sm.values), **extra)


rows, d, delta, resp = [], {}, {}, {}
for v in MODELS:
    x = load(v)
    if x is None:
        print(v, "incomplete", file=sys.stderr)
        continue
    d[v] = x
    perm = x[PP].mean(axis=1)
    delta[v], resp[v] = x.real - perm, (x.real > THR) | (perm > THR)
    fam = "base" if v == "spectrogram" else "ii"
    rows.append(summarise(delta[v][resp[v]], f"{v} responsive", family=fam))
    rows.append(summarise(delta[v], f"{v} all", family="all"))

cs = [v for v in ("spectrogram", "gpt2_current", "gpt2_previous", "whisper", "whisper_untrained") if v in delta]
if len(cs) == 5:
    C = np.logical_or.reduce([resp[v].to_numpy() for v in cs])
    idx = delta["spectrogram"].index[C]
    for v in cs:
        rows.append(summarise(delta[v].loc[idx], f"{v} (common set)", family="common"))
    for a in ("gpt2_current", "gpt2_previous", "whisper_untrained"):
        rows.append(summarise((delta[a] - delta["whisper"]).loc[idx], f"{a} minus whisper (common set)", family="iii"))
    for v in ("gpt2_current", "gpt2_previous", "whisper", "whisper_untrained"):
        rows.append(summarise((delta["spectrogram"] - delta[v]).loc[idx], f"spectrogram minus {v} (common set)", family="descriptive"))

out = pd.DataFrame(rows)
for fam in ("ii", "iii"):
    m = out.family == fam
    if m.sum():
        out.loc[m, "p_holm"] = common.holm(out.loc[m, "p_two_sided"].values)
out.to_csv(common.OUT / "deep_models.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

r = out.set_index("label")


def sig(label):
    return label in r.index and r.loc[label, "p_holm"] < 0.05 and r.loc[label, "subj_median"] > 0


if "gpt2_current minus whisper (common set)" in r.index:
    gph, w, diff = sig("gpt2_current responsive"), sig("whisper responsive"), sig("gpt2_current minus whisper (common set)")
    if gph and not w and diff:
        print("Rule (b): title retained (beyond GPT-2 but not detectably beyond Whisper)")
    elif not gph:
        print("Rule (b): GPT-2 matched in timing/size also carries the information; drop Whisper-specific framing")
    else:
        print(f"Rule (b): detectable beyond GPT-2={gph}, beyond Whisper={w}, tested difference={diff}; no tested difference claim")
    print("Rule (c):", "specific to trained Whisper" if sig("whisper_untrained responsive") and sig("whisper_untrained minus whisper (common set)") else "not met")
