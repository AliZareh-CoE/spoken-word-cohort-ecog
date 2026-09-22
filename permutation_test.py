"""The test the paper rests on: the real cohort values against 205 matched shuffles of the same story.

Each of the 206 assignments is scored against the mean of the other 205, on the electrodes the shuffled models predict
above 0.05. The statistic is the median over patients of each patient's median electrode. The one-sided p value is the
share of assignments scoring at least as high as the real one.

Writes results/permutation_test.csv (the test), permutation_per_patient.csv (per patient) and contribution_size.csv (the size).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

SH = common.OUT / "fits"
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
RA = "_pen7x60_ev50_onsets"
N = 205
THR = 0.05


def load(cond):
    paths = [SH / f"spectrogram_{cond}{RA}_{s}.parquet" for s in SUBS]
    assert all(p.exists() for p in paths), cond
    return pd.concat([pd.read_parquet(p) for p in paths]).set_index(["subject", "electrode"]).cv_r


def d_matrix(allfits: pd.DataFrame) -> pd.DataFrame:
    """D_j = CVR_j - mean of the other assignments, per electrode (columns = assignments, the real one named 'real')."""
    n_other = allfits.shape[1] - 1
    total = allfits.sum(axis=1)
    return allfits.apply(lambda col: col - (total - col) / n_other)


def perm_p(D: pd.DataFrame, subj, sel, how: str) -> dict:
    """Rank the real group statistic among the shuffled ones. how = 'median' (median of patient medians) or 'mean'."""
    g = D[sel].groupby(np.asarray(subj)[np.asarray(sel)])
    pm = g.median() if how == "median" else g.mean()
    t = pm.median() if how == "median" else pm.mean()
    t_real, t_perm = t["real"], t.drop("real")
    n = t_perm.size
    ge, le = int((t_perm >= t_real).sum()), int((t_perm <= t_real).sum())
    p_one = (1 + ge) / (n + 1)
    return dict(T_real=float(t_real), perm_mean=float(t_perm.mean()), perm_sd=float(t_perm.std(ddof=1)), perm_max=float(t_perm.max()),
                n_perm_ge_real=ge, p_one_sided=p_one, p_two_sided=min(1.0, 2 * min(p_one, (1 + le) / (n + 1))))


if __name__ == "__main__":
    real = load("real")
    perm = pd.DataFrame({k: load(f"matched{k}") for k in range(1, N + 1)})
    assert perm.shape[1] == N and perm.index.equals(real.index)
    allfits = pd.concat([real.rename("real"), perm], axis=1)
    subj = allfits.index.get_level_values("subject")
    sym = (perm.mean(axis=1) > THR).to_numpy()
    D = d_matrix(allfits)

    rows = [dict(statistic=name, n_elec=int(sym.sum()), **perm_p(D, subj, sym, how))
            for name, how in (("median of patient medians (primary)", "median"), ("mean of patient means (secondary)", "mean"))]
    out = pd.DataFrame(rows)
    out.to_csv(common.OUT / "permutation_test.csv", index=False)

    pm = D[sym].groupby(np.asarray(subj)[sym]).median()
    pat = pd.DataFrame({"n_elec": pd.Series(sym, index=subj).groupby(level=0).sum(), "real_median": pm["real"],
                        "perm_mean": pm.drop(columns="real").mean(axis=1), "perm_sd": pm.drop(columns="real").std(axis=1, ddof=1),
                        "rank_of_real_from_top": (pm.drop(columns="real").ge(pm["real"], axis=0)).sum(axis=1) + 1})
    pat.to_csv(common.OUT / "permutation_per_patient.csv")

    est = []
    for label, sel in (("original rule (real or mean shuffled CVR > 0.05)", ((real > THR) | (perm.mean(axis=1) > THR)).to_numpy()),
                       ("symmetric set (mean shuffled CVR > 0.05)", sym)):
        d = (real - perm.mean(axis=1))[sel]
        sm = d.groupby(np.asarray(subj)[sel]).median()
        lo, hi = common.boot_ci(sm.values)
        est.append(dict(electrode_rule=label, n_elec=int(sel.sum()), subj_median=float(sm.median()), ci_lo=lo, ci_hi=hi,
                        n_subj_pos=int((sm > 0).sum()), n_subj=int(sm.size), p_wilcoxon=common.wilcoxon_two_sided(sm.values)))
    pd.DataFrame(est).to_csv(common.OUT / "contribution_size.csv", index=False)

    pd.set_option("display.width", 250)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(pat.to_string(float_format=lambda v: f"{v:.4g}"))
    print(pd.DataFrame(est).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    p = out.iloc[0].p_one_sided
    print("Rule:", "p < 0.05 -> the contribution stands under a stimulus-level permutation test (primary evidence)" if p < 0.05
          else "p >= 0.05 -> not distinguishable from the variation between shuffles; the positive claim is withdrawn (negative result, no claim of absence)")
