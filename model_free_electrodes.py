"""The cohort contribution on electrodes chosen without the encoding model, by the ratio of high-gamma variance during
speech to that during the silence before it.

Writes results/model_free_electrodes.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
PP = [f"matched{k}" for k in range(1, 6)]
RA = "_pen7x60_ev50_onsets"


def load(variant, tag):
    fr = {}
    for c in ["real"] + PP:
        paths = [SH / f"{variant}_{c}{tag}_{s}.parquet" for s in SUBS]
        assert all(p.exists() for p in paths), (variant, c)
        fr[c] = pd.concat([pd.read_parquet(p) for p in paths]).set_index(["subject", "electrode"]).cv_r
    return pd.DataFrame(fr)


def summarise(x, label, **extra):
    sm = x.groupby("subject").median()
    lo, hi = common.boot_ci(sm.values)
    return dict(label=label, n_elec=int(x.size), n_subj=int(sm.size), subj_median=float(sm.median()), ci_lo=lo, ci_hi=hi,
                n_subj_pos=int((sm > 0).sum()), p_two_sided=common.wilcoxon_two_sided(sm.values), **extra)


snr = pd.read_parquet(common.MAIN / "data" / "results" / "selection_hg_snr.parquet").set_index(["subject", "electrode"])
selected = snr["selected"].astype(bool)

rows = []
for v in ("spectrogram", "linguistic"):
    d = load(v, RA)
    delta = d.real - d[PP].mean(axis=1)
    s_ = selected.reindex(delta.index).fillna(False).astype(bool)
    rows.append(summarise(delta[s_], f"{v} contribution on model-free electrodes",
                          role="test" if v == "spectrogram" else "descriptive"))
    rows.append(dict(label=f"{v} real median CVR on HG-SNR-selected electrodes",
                     subj_median=float(d.real[s_].groupby("subject").median().median()), role="descriptive"))

out = pd.DataFrame(rows)
out.to_csv(common.OUT / "model_free_electrodes.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
r = out.iloc[0]
print("Rule:", "spectrogram significant on model-free selection -> contribution does not depend on model-based selection"
      if (r.p_two_sided < 0.05 and r.subj_median > 0)
      else "spectrogram not significant on model-free selection -> contribution detected only on electrodes selected by prediction accuracy; report both")
