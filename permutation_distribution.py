"""The statistic of every one of the 206 assignments, for the whole group and for each patient, as the figures need it.
Reproduces results/permutation_test.csv before writing.

Writes results/permutation_distribution_group.csv and permutation_distribution_per_patient.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from permutation_test import N, THR, d_matrix, load

real = load("real")
perm = pd.DataFrame({k: load(f"matched{k}") for k in range(1, N + 1)})
allfits = pd.concat([real.rename("real"), perm], axis=1)
subj = np.asarray(allfits.index.get_level_values("subject"))
sym = (perm.mean(axis=1) > THR).to_numpy()
D = d_matrix(allfits)
pm = D[sym].groupby(subj[sym]).median()
pmean = D[sym].groupby(subj[sym]).mean()
grp = pd.DataFrame({"assignment": [str(c) for c in pm.columns], "is_real": [c == "real" for c in pm.columns],
                    "T_median_of_patient_medians": pm.median().to_numpy(), "T_mean_of_patient_means": pmean.mean().to_numpy()})
ref = pd.read_csv(common.OUT / "permutation_test.csv")
assert abs(grp.loc[grp.is_real, "T_median_of_patient_medians"].iloc[0] - ref.iloc[0].T_real) < 1e-15
assert abs(grp.loc[~grp.is_real, "T_median_of_patient_medians"].max() - ref.iloc[0].perm_max) < 1e-15
assert abs(grp.loc[grp.is_real, "T_mean_of_patient_means"].iloc[0] - ref.iloc[1].T_real) < 1e-15
grp.to_csv(common.OUT / "permutation_distribution_group.csv", index=False)
pat = pm.T.rename_axis("assignment").reset_index()
pat["assignment"] = pat["assignment"].astype(str)
pat.to_csv(common.OUT / "permutation_distribution_per_patient.csv", index=False)
print(grp.describe().round(7).to_string()); print("patients x assignments:", pm.shape)
