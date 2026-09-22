import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_encoding_model as r

C = r.COHORT3


def load():
    ph = pd.read_parquet(Path(__file__).resolve().parents[1] / "results" / "features" / "phonemes_realigned.parquet")
    return ph.dropna(subset=["phoneme_surprisal_in_cohort"]).reset_index(drop=True)


def type_table(ph):
    s = ph.sort_values(["word_idx", "position_in_word"], kind="stable")
    wtype = s.groupby("word_idx")["phoneme_arpabet"].agg(" ".join)
    return s, wtype


def test_word_type_shuffle():
    ph = load()
    p1 = r.permute_word_types(ph, C, 1)
    other = [c for c in ph.columns if c not in C]
    pd.testing.assert_frame_equal(p1[other], ph[other])
    pd.testing.assert_frame_equal(p1, r.permute_word_types(ph, C, 1))
    assert not p1[C].equals(r.permute_word_types(ph, C, 2)[C])
    s, wtype = type_table(p1)
    s = s.assign(wtype=s["word_idx"].map(wtype))
    assert (s.groupby(["wtype", "position_in_word"])[C].nunique() <= 1).all().all()
    s0, wtype0 = type_table(ph)
    s0 = s0.assign(wtype=s0["word_idx"].map(wtype0))
    first = s0.drop_duplicates(["wtype", "position_in_word"]).groupby("wtype")[C].apply(lambda d: tuple(map(tuple, d.to_numpy().round(9))))
    new = s.drop_duplicates(["wtype", "position_in_word"]).groupby("wtype")[C].apply(lambda d: tuple(map(tuple, d.to_numpy().round(9))))
    ntok = wtype0.value_counts()
    info = pd.DataFrame({"len": first.map(len), "bin": pd.cut(ntok.reindex(first.index), [0, 1, 3, 9, np.inf], labels=False)})
    for _, g in info.groupby(["len", "bin"]):
        assert sorted(first[g.index]) == sorted(new[g.index])
    changed = float((first != new).mean())
    assert changed > 0.6, changed


def test_real_values_are_consistent_across_tokens():
    """Premise of A31: in the real data, tokens of a word type share their cohort values (a few exceptions)."""
    ph = load()
    s, wtype = type_table(ph)
    s = s.assign(wtype=s["word_idx"].map(wtype))
    bad = (s.groupby(["wtype", "position_in_word"])[C].nunique() > 1).any(axis=1).mean()
    assert bad < 0.01, bad
