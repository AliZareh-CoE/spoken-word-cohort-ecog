"""Where the cohort contribution lies: by cortical region and hemisphere, measured against 205 shuffled sets. Descriptive,
no test.

Writes results/anatomy_by_region.csv and anatomy_per_electrode.csv.
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
x = pd.DataFrame({"contribution": real - perm, "real": real, "selected": (real > 0.05) | (perm > 0.05)}).reset_index()
est = pd.read_csv(common.OUT / "contribution_size.csv").iloc[0]
assert int(x.selected.sum()) == int(est.n_elec) and abs(x[x.selected].groupby("subject").contribution.median().median() - est.subj_median) < 1e-12
el = pd.read_parquet(common.MAIN / "data" / "derived" / "electrodes.parquet").rename(columns={"name": "electrode"})
x = x.merge(el[["subject", "electrode", "hemisphere", "is_lang_roi", "is_stg_or_mtg", "ho_label"]], on=["subject", "electrode"], how="left")
x["region"] = "other cortex"
x.loc[x.is_lang_roi.fillna(False).astype(bool), "region"] = "other language ROI"
x.loc[x.is_stg_or_mtg.fillna(False).astype(bool), "region"] = "STG/MTG"
x.loc[x.ho_label.isna() | x.ho_label.eq("Background"), "region"] = "no cortical label"
r = x[x.selected]


def desc(g, by):
    return g.groupby(by).agg(n_elec=("contribution", "size"), n_patients=("subject", "nunique"), elec_median_e3=("contribution", lambda v: v.median() * 1e3),
                             share_pos=("contribution", lambda v: (v > 0).mean()), median_real_cvr=("real", "median")).reset_index().rename(columns={by: "level"}).assign(factor=by)


out = pd.concat([desc(r, "region"), desc(r, "hemisphere"), desc(r.assign(all="all selected electrodes"), "all")])[["factor", "level", "n_elec", "n_patients", "elec_median_e3", "share_pos", "median_real_cvr"]]
out.to_csv(common.OUT / "anatomy_by_region.csv", index=False)
x[["subject", "electrode", "contribution", "real", "selected", "region", "hemisphere"]].to_csv(common.OUT / "anatomy_per_electrode.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("unmatched electrodes:", int(r.hemisphere.isna().sum()))
