"""The cohort contribution of the spectrogram model against five shuffled sets, on speech-responsive electrodes and on all
electrodes, with the planted-signal check.

Writes results/cohort_contribution.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
PP = [f"matched{k}" for k in range(1, 6)]
THR = 0.05


def load(variant, tag):
    fr = {}
    for c in ["real"] + PP:
        if tag.startswith("_R"):
            paths = [SH / f"{variant}_{c}{tag}_{s}.parquet" for s in SUBS]
        else:
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


rows = []
LF = "_pen7x60_ev50_onsets"
base = {v: load(v, LF) for v in ("linguistic", "spectrogram")}
bdelta = {v: d.real - d[PP].mean(axis=1) for v, d in base.items()}
bresp = {v: (d.real > THR) | (d[PP].mean(axis=1) > THR) for v, d in base.items()}
for v in ("linguistic", "spectrogram"):
    rows.append(summarise(bdelta[v][bresp[v]], f"{v} responsive", part="main"))
    rows.append(summarise(bdelta[v], f"{v} all", part="all"))
    rows.append(dict(label=f"{v} real median CVR responsive", subj_median=float(base[v].real[bresp[v]].groupby("subject").median().median()), part="descriptive"))
mde = None
for rho in ("0.02", "0.04"):
    d = load("spectrogram", f"_R{rho}_pen7x60_ev50_plantstrata_onsets")
    if d is None:
        print("planted", rho, "incomplete")
        continue
    dplant = d.real - d[PP].mean(axis=1)
    sel = bresp["spectrogram"]
    tot = summarise(dplant[sel], f"spectrogram planted total rho {rho}", part="sensitivity")
    inc = summarise((dplant - bdelta["spectrogram"])[sel], f"spectrogram planted increment rho {rho}", part="sensitivity")
    rows += [tot, inc]
    if mde is None and tot["p_two_sided"] < 0.05 and tot["subj_median"] > 0 and inc["n_subj_pos"] >= 8:
        mde = (rho, inc["subj_median"])
out = pd.DataFrame(rows)
m = out.part == "main"
out.loc[m, "p_holm"] = common.holm(out.loc[m, "p_two_sided"].values)
out.to_csv(common.OUT / "cohort_contribution.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
r = out.set_index("label").loc["spectrogram responsive"]
if r.p_holm < 0.05 and r.subj_median > 0:
    print("Rule (a): cohort contribution beyond spectrogram detected with measured onsets")
elif mde is not None and mde[1] <= 0.19e-3:
    print(f"Rule (b): informative null; MDE {mde[1] * 1e3:.3f}e-3 at rho {mde[0]}")
else:
    print("Rule (b): uninformative null", "(no rho qualified)" if mde is None else f"(MDE {mde[1] * 1e3:.3f}e-3 > 0.19e-3)")
