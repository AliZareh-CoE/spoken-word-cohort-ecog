"""Shared helpers: paths, per-patient medians, bootstrap intervals over patients, the exact Wilcoxon test and the Holm correction.
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

MAIN = Path(os.environ.get("PODCAST_ROOT", Path(__file__).resolve().parent))
RES = MAIN / "data" / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def subject_medians(df, value_col, subject_col="subject"):
    return df.groupby(subject_col)[value_col].median()


def boot_ci(values, stat=np.median, B=20000, seed=0, level=0.95):
    """Percentile bootstrap over subjects."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    boots = np.apply_along_axis(stat, 1, v[idx])
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def wilcoxon_two_sided(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.all(v == 0):
        return float("nan")
    return float(wilcoxon(v, alternative="two-sided").pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (k - i) * p[idx]))
        adj[idx] = running
    return adj


def absorption(uP_ling_subj_medians, uP_mel_subj_medians):
    """1 - median_subj(uP_mel) / median_subj(uP_ling)."""
    return 1.0 - np.median(uP_mel_subj_medians) / np.median(uP_ling_subj_medians)
