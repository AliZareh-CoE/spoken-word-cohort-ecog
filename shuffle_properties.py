"""What the matched shuffle exchanges and what it leaves in place, computed from the table of speech sounds alone, with no
neural data.

Writes results/shuffle_facts.csv and shuffle_variance.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import fit_encoding_model as r
import common

sys.path.insert(0, str(common.MAIN))
C = r.COHORT3
ph = pd.read_parquet(common.OUT / "features" / "phonemes_realigned.parquet")
n_all = len(ph)
ph = ph.dropna(subset=["phoneme_surprisal_in_cohort"]).reset_index(drop=True)
ph = ph.assign(log_cohort=np.log10(ph.cohort_size_before.fillna(1).clip(lower=1)))
V = {"cohort surprisal": "phoneme_surprisal_in_cohort", "log cohort size": "log_cohort", "distance from uniqueness point": "distance_from_uniqueness_point"}
key = r.strata_phoneme(ph)
rows = []
perms = [r.permute_within_strata(ph, C, k, key_fn=r.strata_phoneme) for k in range(1, 6)]
for p_ in perms:
    p_["log_cohort"] = np.log10(p_.cohort_size_before.fillna(1).clip(lower=1))
init = ph.position_in_word == 0
for name, col in V.items():
    x = ph[col].astype(float)
    within = x - x.groupby(key).transform("mean")
    rows.append(dict(variable=name, share_of_variance_within_strata=float(within.var() / x.var()),
                     share_within_strata_word_initial=float((within[init]).var() / x[init].var()) if x[init].var() > 0 else 0.0,
                     corr_real_vs_shuffled=float(np.mean([np.corrcoef(x, p_[col].astype(float))[0, 1] for p_ in perms]))))
facts = pd.DataFrame(rows)
same = np.mean([np.all([np.isclose(ph[c].astype(float), p_[c].astype(float)) for c in C], axis=0).mean() for p_ in perms])
size = key.map(key.value_counts())
from model import features as rc
words, _ = rc.load_features()
fcol = [c for c in words.columns if "freq" in c.lower()]
extra = dict(n_phonemes_in_alignment_file=n_all, n_phonemes_in_models=len(ph), share_tokens_all_three_values_unchanged=float(same),
             share_tokens_in_singleton_strata=float((size == 1).mean()), n_strata=int(key.nunique()), share_word_initial=float(init.mean()), word_freq_column=str(fcol))
if fcol:
    wf = ph.word_idx.map(words.set_index("word_idx")[fcol[0]]).astype(float)
    ok = wf.notna()
    wfw = (wf - wf.groupby(key).transform("mean"))[ok]
    for name, col in V.items():
        x = ph[col].astype(float)
        extra[f"corr_within_stratum_{name}_vs_word_logfreq"] = float(np.corrcoef((x - x.groupby(key).transform("mean"))[ok], wfw)[0, 1])
on = np.sort(ph.phoneme_onset_sec.to_numpy())
ioi = np.diff(on)
bins = np.round(ph.sort_values("phoneme_onset_sec").phoneme_onset_sec.to_numpy() * 100).astype(int)
extra.update(share_inter_onset_intervals_below_50ms=float((ioi < 0.05).mean()), share_events_sharing_a_10ms_bin_with_previous=float((np.diff(bins) == 0).mean()))
ra = ph[ph.phoneme_alignment_method == "charsiu-word"].sort_values(["word_idx", "position_in_word"])
first = ra.groupby("word_idx").first()
w = words.set_index("word_idx")
wcol = [c for c in w.columns if "onset" in c.lower()][0]
extra["share_realigned_words_first_phoneme_at_transcript_word_onset"] = float(np.isclose(first.phoneme_onset_sec, w.loc[first.index, wcol], atol=0.005).mean())
facts.to_csv(common.OUT / "shuffle_variance.csv", index=False)
pd.Series(extra).to_csv(common.OUT / "shuffle_facts.csv", header=["value"])
print(facts.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(pd.Series(extra).to_string())
