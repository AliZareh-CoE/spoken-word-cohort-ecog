import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_encoding_model as r


def _fit_once():
    from model.banded_ridge import fit_clean_banded_ridge
    rng = np.random.default_rng(0)
    T = 3000
    blocks = {"wide": rng.normal(size=(T, 1100)).astype(np.float32), "ev": (rng.random((T, 3)) < 0.05).astype(np.float32)}
    Y = rng.normal(size=(T, 4)).astype(np.float32)
    return fit_clean_banded_ridge(blocks, {"wide": 20, "ev": None}, Y, [0, 50, 100], 100.0, n_folds=3, n_iter=4)["cv_r_mean"]


def test_deterministic_pca_patch():
    r.patch_deterministic_pca()
    a = _fit_once()
    b = _fit_once()
    assert np.array_equal(a, b)


def test_planted_signal_peaks_after_events():
    class RC:
        pass
    n_t, fs = 2000, 100.0
    phon = pd.DataFrame({"phoneme_onset_sec": [5.0], "phoneme_surprisal_in_cohort": [2.0], "cohort_size_before": [100.0],
                         "distance_from_uniqueness_point": [-1.0]})
    phon = pd.concat([phon, phon.assign(phoneme_onset_sec=12.0, phoneme_surprisal_in_cohort=1.0, cohort_size_before=10.0,
                                        distance_from_uniqueness_point=0.0)], ignore_index=True)
    s = r.planted_signal(RC(), phon, n_t, fs)
    assert abs(s.mean()) < 1e-6 and abs(s.std() - 1) < 1e-4
    seg = s[500:560]
    assert int(np.argmax(np.abs(seg - np.median(s)))) in range(13, 18)


def test_entropy_only_permutation_scope():
    phon = pd.read_parquet(str(Path(__file__).resolve().parents[1] / "results/features/phonemes_hybrid_with_entropy.parquet"))
    phon = phon.dropna(subset=["phoneme_surprisal_in_cohort"]).reset_index(drop=True)
    p = r.permute_within_strata(phon, [r.ENT], 1)
    other = [c for c in phon.columns if c != r.ENT]
    pd.testing.assert_frame_equal(p[other], phon[other])
    assert (p[r.ENT].to_numpy() != phon[r.ENT].to_numpy()).mean() > 0.5


def test_melu_uses_untrained_whisper_same_shape():
    """A18: whisper_untrained differs from whisper only in the whisper block, which is the untrained encoder with the same shape."""
    from model import features as rc
    assert r.FAMS["whisper_untrained"] == r.FAMS["whisper"]
    assert r.UNTRAINED_WHISPER.exists()
    n_t = 2000
    trained = rc.load_whisper_raw(n_t)
    untrained = rc.load_whisper_raw(n_t, r.UNTRAINED_WHISPER)
    assert trained.shape == untrained.shape == (n_t, 1024)
    assert not np.allclose(trained, untrained)


def test_pperm_preserves_phoneme_position_length_strata():
    """A19: matched keeps every token's (position, length, phoneme) stratum and the within-stratum multiset of cohort values."""
    from model import features as rc
    _, phon = rc.load_features()
    out = r.permute_within_strata(phon, r.COHORT3, 1, key_fn=r.strata_phoneme)
    key = r.strata_phoneme(phon)
    assert (r.strata_phoneme(out) == key).all()
    assert (out.phoneme_arpabet.values == phon.phoneme_arpabet.values).all()
    for c in r.COHORT3:
        a = phon.groupby(key)[c].apply(lambda x: sorted(x.fillna(-1e9)))
        b = out.groupby(key)[c].apply(lambda x: sorted(x.fillna(-1e9)))
        assert (a == b).all()
    assert not np.allclose(out[r.COHORT3[0]].fillna(0).values, phon[r.COHORT3[0]].fillna(0).values)
    j = phon.set_index("phoneme_idx")[r.COHORT3].fillna(-1).apply(tuple, axis=1)
    k = out.set_index("phoneme_idx")[r.COHORT3].fillna(-1).apply(tuple, axis=1)
    assert sorted(j) == sorted(k)


def test_boxcar_tiles_50ms_lags():
    """A22: an impulse spread over 5 bins, lagged every 5 bins, covers every bin of the response window."""
    X = np.zeros((200, 1), dtype=np.float32)
    X[10, 0] = 2.0
    B = r.boxcar(X, 5)
    assert np.allclose(B[10:15, 0], 2.0) and B[:10].sum() == 0 and B[15:].sum() == 0
    covered = np.zeros(200, bool)
    for lag in range(0, 101, 5):
        covered[10 + lag:15 + lag] |= True
    assert covered[10:115].all()
    assert r.boxcar(X, 1) is X


def test_gpt_phoneme_series_places_word_embedding_at_each_phoneme():
    """A22: gpt2_current block repeats the current word's embedding at each of its phoneme onsets."""
    words = pd.DataFrame({"word_idx": [0, 1], "onset_sec": [0.10, 0.50]})
    phon = pd.DataFrame({"word_idx": [0, 0, 1], "phoneme_onset_sec": [0.10, 0.20, 0.50]})
    emb = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = r.gpt_phoneme_series(words, phon, emb, np.array([0, 1]), 100, 100.0)
    assert np.allclose(out[10], [1, 2]) and np.allclose(out[20], [1, 2]) and np.allclose(out[50], [3, 4])
    assert np.count_nonzero(out.any(axis=1)) == 3


def test_cohort_band_columns_are_the_three_cohort_variables():
    """A23: COHORT_COLS pick exactly cohort surprisal, log cohort size and UP distance from the phoneme family."""
    from model import features as rc
    _, phon = rc.load_features()
    names = [f.name for f in rc.make_phon_features(phon)]
    assert [names[j] for j in r.COHORT_COLS] == ["cohort_surp", "log_cohort", "dist_from_UP"]


def test_planted_signal_phoneme_strata_residualizes_within_identity(monkeypatch):
    """A23: with phoneme strata, each residualized cohort variable has zero mean within every position x length x phoneme stratum."""
    from model import features as rc
    _, phon = rc.load_features()
    monkeypatch.setenv("PLANTED_SIGNAL_STRATA", "phoneme")
    assert r.plant_phoneme_strata()
    ph = phon.assign(_l=np.log10(phon.cohort_size_before.fillna(1).clip(lower=1)))
    key = r.strata_phoneme(ph)
    res = ph["_l"] - ph.groupby(key)["_l"].transform("mean")
    assert np.abs(res.groupby(key).mean()).max() < 1e-9


def test_lowfreq_loader_matches_highgamma_channels_and_is_low_frequency():
    """A24: low-frequency response has the high-gamma channels and length, with >90% of power below 10 Hz."""
    from model import features as rc
    Y, names, fs = r.load_lowfreq(rc, "sub-08")
    hg, hg_names, hg_fs = rc.load_hg_raw("sub-08")
    assert names == hg_names and Y.shape == hg.shape and fs == hg_fs
    x = Y[:, : min(10, Y.shape[1])] - Y[:, : min(10, Y.shape[1])].mean(axis=0)
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], 1 / fs)
    assert spec[freqs < 10].sum() / spec.sum() > 0.9


def test_realigned_phonemes_change_only_onsets_of_accepted_words():
    """A25: re-aligned table keeps rows and cohort values; only accepted words change timing, monotonic and within tolerance."""
    import pytest
    path = r.HERE / "results" / "features" / "phonemes_realigned.parquet"
    if not path.exists():
        pytest.skip("re-alignment not yet run")
    new = pd.read_parquet(path)
    old = pd.read_parquet(r.MAIN / "data" / "features" / "phonemes_hybrid.parquet")
    assert len(new) == len(old) and (new.phoneme_idx.values == old.phoneme_idx.values).all()
    for c in ("phoneme_surprisal_in_cohort", "cohort_size_before", "distance_from_uniqueness_point", "phoneme_arpabet", "word_idx", "position_in_word"):
        assert new[c].equals(old[c])
    changed = new.phoneme_onset_sec.values != old.phoneme_onset_sec.values
    assert (new.phoneme_alignment_method[changed] == "charsiu-word").all()
    assert (old.phoneme_alignment_method[new.phoneme_alignment_method == "charsiu-word"] != "charsiu").all()
    words = pd.read_parquet(r.MAIN / "data" / "features" / "words.parquet").set_index("word_idx")
    acc = new[new.phoneme_alignment_method == "charsiu-word"]
    for wi, g in acc.groupby("word_idx"):
        on = g.sort_values("phoneme_idx").phoneme_onset_sec.to_numpy()
        assert (np.diff(on) > 0).all()
        assert on.min() >= words.loc[wi, "onset_sec"] - 0.05 - 1e-9 and on.max() <= words.loc[wi, "offset_sec"] + 0.05 + 1e-9
