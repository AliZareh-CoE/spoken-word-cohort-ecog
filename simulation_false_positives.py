"""False-positive rates on simulated data with shared stimulus noise and no true effect, for the permutation test and for
a test across patients. Explains why the paper uses the permutation test.

Writes results/simulation_false_positives.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common
from permutation_test import d_matrix, perm_p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_permutation_test import N_SHUF, sim

rng = np.random.default_rng(0)
n, fp_perm, fp_w = 300, 0, 0
for _ in range(n):
    a, subj = sim(rng, 0.0)
    sel = np.ones(len(a), bool)
    fp_perm += perm_p(d_matrix(a), subj, sel, "median")["p_one_sided"] < 0.05
    sm = (a["real"] - a.drop(columns="real").iloc[:, :5].mean(axis=1)).groupby(subj).median()
    fp_w += common.wilcoxon_two_sided(sm.values) < 0.05
out = pd.DataFrame([dict(n_simulations=n, n_shuffles=N_SHUF, shared_noise_sd_over_electrode_noise_sd=1.0,
                         false_positive_rate_permutation_test=fp_perm / n, false_positive_rate_patient_level_wilcoxon=fp_w / n)])
out.to_csv(common.OUT / "simulation_false_positives.csv", index=False)
print(out.to_string(index=False))
