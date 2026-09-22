"""The cohort contribution as a share of the prediction accuracy of the model, per patient, against 205 shuffled sets.

Writes results/effect_size.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
RA = "_pen7x60_ev50_onsets"


def load(c):
    return pd.concat([pd.read_parquet(SH / f"spectrogram_{c}{RA}_{s}.parquet") for s in SUBS]).set_index(["subject", "electrode"]).cv_r


real = load("real")
perm = pd.DataFrame({k: load(f"matched{k}") for k in range(1, 206)}).mean(axis=1)
resp = (real > 0.05) | (perm > 0.05)
g = pd.DataFrame({"real_cvr": real[resp], "contribution": (real - perm)[resp]}).groupby("subject").median()
g["ratio"] = g.contribution / g.real_cvr
est = pd.read_csv(common.OUT / "contribution_size.csv").iloc[0]
assert abs(g.contribution.median() - est.subj_median) < 1e-12 and int(resp.sum()) == int(est.n_elec)
out = pd.DataFrame([dict(quantity="electrodes (real or mean of 205 shuffles > 0.05)", value=int(resp.sum())),
                    dict(quantity="median across patients of patient-median real CVR", value=float(g.real_cvr.median())),
                    dict(quantity="median across patients of patient-median contribution (205-shuffle baseline)", value=float(g.contribution.median())),
                    dict(quantity="median across patients of contribution / real CVR", value=float(g.ratio.median())),
                    dict(quantity="min ratio", value=float(g.ratio.min())), dict(quantity="max ratio", value=float(g.ratio.max()))])
out.to_csv(common.OUT / "effect_size.csv", index=False)
print(g.round(6).to_string()); print(out.to_string(index=False))
