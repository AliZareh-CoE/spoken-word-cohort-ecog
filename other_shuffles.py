"""The cohort contribution under two other shuffles: values exchanged among speech sounds at the same position in words of
the same length, and a shuffle that gives every repetition of a word the same values.

Writes results/other_shuffles.csv.
"""
from __future__ import annotations

import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
THR = 0.05
RA = "_pen7x60_ev50_onsets"


def load(conds):
    fr = {}
    for c in conds:
        paths = [SH / f"spectrogram_{c}{RA}_{s}.parquet" for s in SUBS]
        assert all(p.exists() for p in paths), c
        fr[c] = pd.concat([pd.read_parquet(p) for p in paths]).set_index(["subject", "electrode"]).cv_r
    return pd.DataFrame(fr)


def summarise(x, label, **extra):
    sm = x.groupby("subject").median()
    lo, hi = common.boot_ci(sm.values)
    return dict(label=label, n_elec=int(x.size), n_subj=int(sm.size), subj_median=float(sm.median()), ci_lo=lo, ci_hi=hi,
                n_subj_pos=int((sm > 0).sum()), p_two_sided=common.wilcoxon_two_sided(sm.values), **extra)


W, S, P = [f"word_repeat{k}" for k in range(1, 6)], [f"position_length{k}" for k in range(1, 6)], [f"matched{k}" for k in range(1, 6)]
d = load(["real"] + W + S + P)
rows, delta, resp = [], {}, {}
for name, conds in (("word-consistent shuffle (word_repeat)", W), ("token-level position x length shuffle (position_length)", S), ("matched shuffle (matched, A25)", P)):
    perm = d[conds].mean(axis=1)
    delta[name], resp[name] = d.real - perm, (d.real > THR) | (perm > THR)
    rows.append(summarise(delta[name][resp[name]], f"spectrogram {name}, responsive", family="test" if "matched" not in name else "reference"))
    rows.append(summarise(delta[name], f"spectrogram {name}, all electrodes", family="all"))
kw, ks = "word-consistent shuffle (word_repeat)", "token-level position x length shuffle (position_length)"
common_set = resp[kw] | resp[ks]
rows.append(summarise((delta[ks] - delta[kw])[common_set], "position_length minus word_repeat contribution (common set)", family="descriptive"))
out = pd.DataFrame(rows)
m = out.family == "test"
out.loc[m, "p_holm"] = common.holm(out.loc[m, "p_two_sided"].values)
out.to_csv(common.OUT / "other_shuffles.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
r = out.set_index("label")
w, s = r.loc[f"spectrogram {kw}, responsive"], r.loc[f"spectrogram {ks}, responsive"]
w_sig, s_sig = (w.p_holm < 0.05 and w.subj_median > 0), (s.p_holm < 0.05 and s.subj_median > 0)
if w_sig:
    print("Rule: contribution significant under the word-consistent shuffle -> it does not depend on shuffled values differing between repetitions of a word")
elif s_sig:
    print("Rule: significant under the token-level shuffle but not under the word-consistent shuffle -> the contribution may reflect the consistency of cohort values across repetitions of a word")
else:
    print("Rule: neither null gives a significant contribution -> comparison uninformative")
