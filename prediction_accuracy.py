"""Prediction accuracy of the spectrogram model and of the models with GPT-2 or Whisper added: per electrode, by region,
and summarised. Descriptive, no test.

Writes results/prediction_accuracy.csv, prediction_accuracy_per_electrode.csv and prediction_accuracy_by_region.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
RA = "_pen7x60_ev50_onsets"
MODELS = [("spectrogram", "spectrogram model"), ("gpt2_current", "+ GPT-2 (current word, every phoneme, 64 components)"), ("whisper", "+ Whisper (64 components)")]

cv = pd.DataFrame({v: pd.concat([pd.read_parquet(SH / f"{v}_real{RA}_{s}.parquet") for s in SUBS]).set_index(["subject", "electrode"]).cv_r for v, _ in MODELS})
resp = cv["spectrogram"] > 0.05
rows = []
for v, lab in MODELS:
    x = cv[v]
    rows.append(dict(model=lab, variant=v, n_electrodes=int(x.size), max_cvr=float(x.max()), n_above_0p05=int((x > 0.05).sum()),
                     median_cvr_on_spectrogram_responsive=float(x[resp].median()), n_spectrogram_responsive=int(resp.sum()),
                     median_gain_over_spectrogram_on_those=float((x - cv["spectrogram"])[resp].median()),
                     share_of_those_improved=float(((x - cv["spectrogram"])[resp] > 0).mean())))
out = pd.DataFrame(rows)
out.to_csv(common.OUT / "prediction_accuracy.csv", index=False)
cv.reset_index().to_csv(common.OUT / "prediction_accuracy_per_electrode.csv", index=False)
el = pd.read_parquet(common.MAIN / "data" / "derived" / "electrodes.parquet").rename(columns={"name": "electrode"})
x = cv["spectrogram"].rename("cv_r").reset_index().merge(el[["subject", "electrode", "is_lang_roi", "is_stg_or_mtg", "ho_label"]], on=["subject", "electrode"], how="left")
x["region"] = "other cortex"
x.loc[x.is_lang_roi.fillna(False).astype(bool), "region"] = "other language ROI"
x.loc[x.is_stg_or_mtg.fillna(False).astype(bool), "region"] = "STG/MTG"
x.loc[x.ho_label.isna() | x.ho_label.eq("Background"), "region"] = "no cortical label"
top = x.nlargest(50, "cv_r")
reg = x.groupby("region").agg(n_elec=("cv_r", "size"), median_cvr=("cv_r", "median"), n_above_0p05=("cv_r", lambda v: int((v > 0.05).sum())), max_cvr=("cv_r", "max"))
reg["share_above_0p05"] = reg.n_above_0p05 / reg.n_elec
reg["n_of_top50"] = top.region.value_counts().reindex(reg.index).fillna(0).astype(int)
reg.reset_index().to_csv(common.OUT / "prediction_accuracy_by_region.csv", index=False)
pd.set_option("display.width", 250)
print(reg.to_string(float_format=lambda v: f"{v:.4g}"))
print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
