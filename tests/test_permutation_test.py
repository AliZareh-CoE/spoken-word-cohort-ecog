import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from permutation_test import d_matrix, perm_p

N_ELEC = [40, 30, 25, 20, 15, 12, 9, 9, 5]
N_SHUF = 60


def sim(rng, effect, shared_sd=1.0, elec_sd=1.0):
    """CVR for 1 real + N_SHUF shuffled assignments. Each assignment has a stimulus-side offset SHARED by all patients
    (chance alignment of that assignment with the stimulus), plus independent electrode noise."""
    subj = np.repeat([f"s{i}" for i in range(9)], N_ELEC)
    shared = rng.normal(0, shared_sd, N_SHUF + 1)
    x = shared[None, :] + rng.normal(0, elec_sd, (subj.size, N_SHUF + 1))
    x[:, 0] += effect
    cols = ["real"] + list(range(1, N_SHUF + 1))
    return pd.DataFrame(x, columns=cols), subj


def test_false_positive_rate_with_shared_stimulus_noise():
    rng = np.random.default_rng(0)
    n, fp_perm, fp_wilcoxon = 300, 0, 0
    for _ in range(n):
        a, subj = sim(rng, 0.0)
        D = d_matrix(a)
        sel = np.ones(len(a), bool)
        fp_perm += perm_p(D, subj, sel, "median")["p_one_sided"] < 0.05
        sm = (a["real"] - a.drop(columns="real").iloc[:, :5].mean(axis=1)).groupby(subj).median()
        fp_wilcoxon += common.wilcoxon_two_sided(sm.values) < 0.05
    assert fp_perm / n <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / n), fp_perm / n
    assert fp_wilcoxon / n > 0.3, fp_wilcoxon / n


def test_detects_planted_effect_and_sign():
    rng = np.random.default_rng(1)
    a, subj = sim(rng, +4.0)
    sel = np.ones(len(a), bool)
    r = perm_p(d_matrix(a), subj, sel, "median")
    assert r["T_real"] > 0 and r["p_one_sided"] < 0.05
    a, subj = sim(rng, -4.0)
    r = perm_p(d_matrix(a), subj, sel, "median")
    assert r["T_real"] < 0 and r["p_one_sided"] > 0.9


def test_d_matrix_is_value_minus_mean_of_others():
    a = pd.DataFrame({"real": [3.0, 0.0], 1: [1.0, 2.0], 2: [2.0, 4.0]})
    D = d_matrix(a)
    assert np.allclose(D["real"], [3 - 1.5, 0 - 3.0]) and np.allclose(D[1], [1 - 2.5, 2 - 2.0])
