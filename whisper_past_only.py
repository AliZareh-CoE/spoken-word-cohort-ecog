"""The cohort contribution with a Whisper restricted to past audio added to the spectrogram model, its control with 500 ms
of later audio, and how much of a planted signal each model recovers.

Writes results/whisper_past_only.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
PP = [f"matched{k}" for k in range(1, 6)]
THR = 0.05
RA = "_pen7x60_ev50_onsets"
PLANT = "_R0.04_pen7x60_ev50_plantstrata_onsets"


def load(variant, tag):
    fr = {}
    for c in ["real"] + PP:
        paths = [SH / f"{variant}_{c}{tag}_{s}.parquet" for s in SUBS]
        if not all(p.exists() for p in paths):
            return None
        fr[c] = pd.concat([pd.read_parquet(p) for p in paths]).set_index(["subject", "electrode"]).cv_r
    return pd.DataFrame(fr)


def summarise(x, label, **extra):
    sm = x.groupby("subject").median()
    lo, hi = common.boot_ci(sm.values)
    return dict(label=label, n_elec=int(x.size), n_subj=int(sm.size), subj_median=float(sm.median()), ci_lo=lo, ci_hi=hi,
                n_subj_pos=int((sm > 0).sum()), p_two_sided=common.wilcoxon_two_sided(sm.values), **extra)


V = ["spectrogram", "whisper", "whisper_past", "whisper_past_500ms"]
d = {v: load(v, RA) for v in V}
assert all(x is not None for x in d.values()), {v: x is not None for v, x in d.items()}
delta = {v: x.real - x[PP].mean(axis=1) for v, x in d.items()}
resp = {v: (x.real > THR) | (x[PP].mean(axis=1) > THR) for v, x in d.items()}

rows = []
for v in ("whisper_past", "whisper_past_500ms"):
    rows.append(summarise(delta[v][resp[v]], f"{v} responsive", family="models"))
    rows.append(summarise(delta[v], f"{v} all", family="all"))
common_set = resp["spectrogram"] | resp["whisper"] | resp["whisper_past"] | resp["whisper_past_500ms"]
for v in V:
    rows.append(summarise(delta[v][common_set], f"{v} (common set)", family="common"))
rows.append(summarise((delta["whisper_past"] - delta["whisper"])[common_set], "whisper_past minus whisper (common set)", family="paired"))
rows.append(summarise((delta["whisper_past_500ms"] - delta["whisper"])[common_set], "whisper_past_500ms minus whisper (common set)", family="paired"))
rows.append(summarise((delta["whisper_past"] - delta["whisper_past_500ms"])[common_set], "whisper_past minus whisper_past_500ms (common set)", family="descriptive"))
rows.append(summarise((delta["spectrogram"] - delta["whisper_past"])[common_set], "spectrogram minus whisper_past (common set)", family="descriptive"))
for v in ("whisper_past", "whisper_past_500ms"):
    rows.append(summarise((d[v].real - d["whisper"].real)[common_set], f"real CVR {v} minus whisper (common set)", family="descriptive"))

ratio = None
pm, pw = load("spectrogram", PLANT), load("whisper_past", PLANT)
if pm is not None and pw is not None:
    sel = resp["spectrogram"] | resp["whisper_past"]
    inc_m = ((pm.real - pm[PP].mean(axis=1)) - delta["spectrogram"])[sel]
    inc_w = ((pw.real - pw[PP].mean(axis=1)) - delta["whisper_past"])[sel]
    rm, rw = summarise(inc_m, "spectrogram planted increment rho 0.04", family="sensitivity"), summarise(inc_w, "whisper_past planted increment rho 0.04", family="sensitivity")
    rows += [rm, rw]
    ratio = rw["subj_median"] / rm["subj_median"]
    rows.append(dict(label="planted ratio whisper_past/spectrogram", subj_median=ratio, family="sensitivity"))
else:
    print("planted shards incomplete")

out = pd.DataFrame(rows)
for fam in ("models", "paired"):
    m = out.family == fam
    out.loc[m, "p_holm"] = common.holm(out.loc[m, "p_two_sided"].values)
out.to_csv(common.OUT / "whisper_past_only.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

r = out.set_index("label")
wc, diff = r.loc["whisper_past responsive"], r.loc["whisper_past minus whisper (common set)"]
wc_sig = wc.p_holm < 0.05 and wc.subj_median > 0
if wc_sig and diff.p_holm < 0.05 and diff.subj_median > 0:
    print("Rule (a): the loss of the cohort contribution with Whisper depends on Whisper hearing the rest of the word")
elif not wc_sig and ratio is not None and ratio >= 0.5:
    print(f"Rule (b): not detectable beyond past-only Whisper, planted ratio {ratio:.2f} >= 0.5 -> look-ahead alone does not explain the loss")
elif not wc_sig and ratio is not None:
    print(f"Rule (c): not detectable beyond past-only Whisper, planted ratio {ratio:.2f} < 0.5 -> uninformative null (lost sensitivity)")
else:
    print("Inconclusive under the pre-set rules (report as such)")
