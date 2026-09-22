"""Fits the banded-ridge encoding model for one feature set, one condition and every patient, and stores each electrode's
cross-validated prediction accuracy in results/fits/.

The condition is real (the real cohort values) or one shuffle of them: matched{k} exchanges values among speech sounds
that share phoneme identity, position in the word and word length; position_length{k} drops phoneme identity; word_repeat{k} gives
every repetition of a word the same values.

Usage: python fit_encoding_model.py --variant spectrogram --cond real [--plant 0.04] [--jobs 9]
The settings of the paper come from the environment: PENALTY_MAX_LOG10=7 PENALTY_CANDIDATES=60 EVENT_WINDOW_50MS=1 PHONEME_ONSETS=realigned.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MAIN = Path(os.environ.get("PODCAST_ROOT", Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "fits"
COHORT3 = ["phoneme_surprisal_in_cohort", "cohort_size_before", "distance_from_uniqueness_point"]
ENT = "cohort_entropy_after"
WHISPER_FILES = {"whisper_past": HERE / "results" / "features" / "whisper_causal.hdf5",
                 "whisper_past_500ms": HERE / "results" / "features" / "whisper_causal_la500.hdf5"}
UNTRAINED_WHISPER = MAIN / "data" / "derived" / "whisper_untrained.hdf5"
FAMS = {
    "linguistic": ["word_sparse", "phon_sparse", "sent"],
    "spectrogram": ["mel", "word_sparse", "phon_sparse", "sent"],
    "whisper": ["mel", "whisper", "word_sparse", "phon_sparse", "sent"],
    "whisper_untrained": ["mel", "whisper", "word_sparse", "phon_sparse", "sent"],
    "whisper_past": ["mel", "whisper", "word_sparse", "phon_sparse", "sent"],
    "whisper_past_500ms": ["mel", "whisper", "word_sparse", "phon_sparse", "sent"],
    "gpt2_word_onset": ["mel", "word_sparse", "gpt2xl_emb", "phon_sparse", "sent"],
    "gpt2_current": ["mel", "word_sparse", "gpt2xl_phon", "phon_sparse", "sent"],
    "gpt2_previous": ["mel", "word_sparse", "gpt2xl_prev", "phon_sparse", "sent"],
    "full": ["mel", "whisper", "word_sparse", "gpt2xl_emb", "phon_sparse", "sent"],
}
ORDER = ["mel", "whisper", "word_sparse", "gpt2xl_emb", "gpt2xl_prev", "gpt2xl_phon", "cohort3", "phon_sparse", "sent"]
EVENT_BLOCKS = {"word_sparse", "gpt2xl_emb", "gpt2xl_prev", "gpt2xl_phon", "cohort3", "phon_sparse", "sent"}
COHORT_COLS = [1, 2, 3]


def penalty_settings():
    """A20: optional wide penalty search via environment (defaults = original settings)."""
    hi = float(os.environ.get("PENALTY_MAX_LOG10", "5"))
    n_iter = int(os.environ.get("PENALTY_CANDIDATES", "30"))
    tag = "" if (hi == 5 and n_iter == 30) else f"_pen{hi:g}x{n_iter}"
    if box_width() > 1:
        tag += "_ev50"
    if cohort_band():
        tag += "_cohortpen"
    if plant_phoneme_strata():
        tag += "_plantstrata"
    if low_freq():
        tag += "_lowfreq"
    if realigned():
        tag += "_onsets"
    return hi, n_iter, tag


def realigned():
    """Use word-level re-aligned phoneme onsets when PHONEME_ONSETS=realigned."""
    return os.environ.get("PHONEME_ONSETS", "") == "realigned"


def load_features(rc):
    words, phon = rc.load_features()
    if realigned():
        import pandas as pd
        phon = pd.read_parquet(HERE / "results" / "features" / "phonemes_realigned.parquet")
        phon = phon.dropna(subset=["phoneme_surprisal_in_cohort"]).reset_index(drop=True)
    return words, phon


def low_freq():
    """A24: use 1-8 Hz low-frequency activity as the response when RESPONSE_BAND=lowfreq."""
    return os.environ.get("RESPONSE_BAND", "") == "lowfreq"


def load_lowfreq(rc, sub):
    """A24: 1-8 Hz band-passed broadband ECoG (zero-phase FIR), resampled to the high-gamma rate, high-gamma channel order.
    Cached per subject under results/features/lowfreq/."""
    import numpy as np
    import mne
    hg, names, fs = rc.load_hg_raw(sub)
    cache = HERE / "results" / "features" / "lowfreq" / f"{sub}_1-8Hz_{int(fs)}Hz.npy"
    if cache.exists():
        Y = np.load(cache)
    else:
        raw = mne.io.read_raw_fif(MAIN / "data" / "ds005574" / "derivatives" / "ecogprep" / sub / "ieeg" / f"{sub}_task-podcast_ieeg.fif",
                                  preload=True, verbose="ERROR")
        raw.pick(names)
        raw.filter(1.0, 8.0, method="fir", phase="zero", verbose="ERROR")
        raw.resample(fs, npad="auto", verbose="ERROR")
        Y = raw.get_data().T.astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, Y)
    assert Y.shape[1] == len(names)
    n = min(Y.shape[0], hg.shape[0])
    if Y.shape[0] < hg.shape[0]:
        Y = np.vstack([Y, np.zeros((hg.shape[0] - Y.shape[0], Y.shape[1]), np.float32)])
    return Y[: hg.shape[0]], names, fs


def cohort_band():
    """A23: the three cohort columns get their own penalty (family 'cohort3') when COHORT_OWN_PENALTY=1."""
    return os.environ.get("COHORT_OWN_PENALTY", "0") == "1"


def plant_phoneme_strata():
    """A23: residualize the planted signal against position x length x phoneme identity when PLANTED_SIGNAL_STRATA=phoneme."""
    return os.environ.get("PLANTED_SIGNAL_STRATA", "") == "phoneme"


def box_width():
    """A22: event window in samples (EVENT_WINDOW_50MS=1 -> 5 samples = 50 ms at 100 Hz); 1 = original single-bin events."""
    return 5 if os.environ.get("EVENT_WINDOW_50MS", "0") == "1" else 1


def boxcar(X, width):
    """Spread each event over its onset bin and the following width-1 bins (causal boxcar)."""
    if width <= 1:
        return X
    out = X.copy()
    for k in range(1, width):
        out[k:] += X[:-k]
    return out


def gpt_phoneme_series(words, phon, emb, wi, n_t, fs):
    """A22: the current word's GPT-2 embedding at every phoneme onset of that word."""
    import numpy as np
    row = {int(w): i for i, w in enumerate(wi)}
    out = np.zeros((n_t, emb.shape[1]), dtype=np.float32)
    for w, t in zip(phon["word_idx"].to_numpy(), phon["phoneme_onset_sec"].to_numpy()):
        i = row.get(int(w))
        k = int(round(t * fs))
        if i is not None and 0 <= k < n_t:
            out[k] = emb[i]
    return out


def shard_name(variant, cond, sub, plant):
    suffix = (f"_R{plant:g}" if plant else "") + penalty_settings()[2]
    return f"{variant}_{cond}{suffix}_{sub}.parquet"


def strata(phon):
    pos = phon["position_in_word"].clip(upper=6).astype(int)
    length = phon.groupby("word_idx")["phoneme_idx"].transform("size").clip(upper=8).astype(int)
    return pos * 10 + length


def strata_phoneme(phon):
    """A19: position x length x phoneme identity (ARPAbet)."""
    code = phon["phoneme_arpabet"].astype("category").cat.codes.astype(int)
    return strata(phon) * 100 + code


def permute_within_strata(phon, cols, seed, key_fn=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = phon.copy()
    key = (key_fn or strata)(phon).to_numpy()
    new = {c: phon[c].to_numpy().copy() for c in cols}
    old = {c: phon[c].to_numpy() for c in cols}
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        perm = rng.permutation(idx)
        for c in cols:
            new[c][idx] = old[c][perm]
    for c in cols:
        out[c] = new[c]
    return out


def permute_word_types(phon, cols, seed):
    """A31: permute cohort values between word types (pronunciations) of equal length and token-count bin.
    Every token of a type receives, position by position, the values of the type assigned to it."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    ph = phon.sort_values(["word_idx", "position_in_word"], kind="stable")
    wtype = ph.groupby("word_idx")["phoneme_arpabet"].agg(" ".join)
    wlen = ph.groupby("word_idx").size()
    tokens_of = {t: g.index.to_numpy() for t, g in wtype.groupby(wtype)}
    ntok = wtype.map(wtype.value_counts())
    info = pd.DataFrame({"wtype": wtype, "len": wlen, "bin": pd.cut(ntok, [0, 1, 3, 9, np.inf], labels=False).astype(int)})
    types = info.drop_duplicates("wtype").set_index("wtype")
    first_tok = {t: toks[0] for t, toks in tokens_of.items()}
    rows_of_word = {w: g.index.to_numpy() for w, g in ph.groupby("word_idx")}
    seq = {t: {c: phon.loc[rows_of_word[first_tok[t]], c].to_numpy() for c in cols} for t in tokens_of}
    out = phon.copy()
    new = {c: phon[c].to_numpy().copy() for c in cols}
    pos_of_row = {r: i for i, r in enumerate(phon.index)}
    for _, grp in types.groupby(["len", "bin"]):
        ts = grp.index.to_numpy()
        src = ts[rng.permutation(len(ts))]
        for t, s_ in zip(ts, src):
            for w in tokens_of[t]:
                rows = rows_of_word[w]
                for c in cols:
                    vals = seq[s_][c]
                    assert len(vals) == len(rows)
                    for r, v in zip(rows, vals):
                        new[c][pos_of_row[r]] = v
    for c in cols:
        out[c] = new[c]
    return out


def patch_deterministic_pca():
    import functools
    from sklearn.decomposition import PCA
    import model.banded_ridge as brc
    brc.PCA = functools.partial(PCA, svd_solver="full")


def phon_features(rc, phon, with_entropy):
    from model.encoding import FeatureSpec
    feats = rc.make_phon_features(phon)
    if with_entropy:
        feats.insert(4, FeatureSpec("cohort_entropy", phon, ENT, "phoneme_onset_sec"))
    return feats


def planted_signal(rc, phon_real, n_t, fs):
    import numpy as np
    import pandas as pd
    from model.encoding import FeatureSpec, build_event_series
    ph = phon_real.assign(_l=np.log10(phon_real.cohort_size_before.fillna(1).clip(lower=1)))
    key = (strata_phoneme(ph) if plant_phoneme_strata() else strata(ph)) if "word_idx" in ph.columns else pd.Series(0, index=ph.index)
    for c in ["phoneme_surprisal_in_cohort", "_l", "distance_from_uniqueness_point"]:
        ph[c + "_r"] = ph[c] - ph.groupby(key)[c].transform("mean")
    feats = [FeatureSpec("s", ph, "phoneme_surprisal_in_cohort_r", "phoneme_onset_sec"),
             FeatureSpec("l", ph, "_l_r", "phoneme_onset_sec"),
             FeatureSpec("d", ph, "distance_from_uniqueness_point_r", "phoneme_onset_sec")]
    X = build_event_series(feats, n_t, fs).astype(np.float64)
    ev = X.sum(axis=1)
    lags = np.arange(0, int(0.4 * fs) + 1)
    kern = np.exp(-0.5 * ((lags / fs - 0.150) / 0.050) ** 2)
    s = np.zeros(n_t)
    for li, w in zip(lags, kern):
        s[li:] += w * ev[: n_t - li]
    return ((s - s.mean()) / (s.std() + 1e-12)).astype(np.float32)


def fit_subject(args):
    variant, cond, sub, plant = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    sys.path.insert(0, str(MAIN))
    import numpy as np
    import pandas as pd
    from model import features as rc
    from model.banded_ridge import fit_clean_banded_ridge
    from model.encoding import DEFAULT_LAGS_MS, build_event_series

    patch_deterministic_pca()
    t0 = time.time()
    kind, base = variant.split("_", 1)
    with_entropy = kind == "E"
    words, phon = load_features(rc)
    if with_entropy:
        pe = pd.read_parquet(HERE / "results" / "features" / "phonemes_hybrid_with_entropy.parquet")[["phoneme_idx", ENT]]
        phon = phon.merge(pe, on="phoneme_idx", how="left")
        assert phon[ENT].notna().all()
    phon_real = phon
    cols = COHORT3 + ([ENT] if with_entropy else [])
    if cond.startswith("entropy_only"):
        phon = permute_within_strata(phon, [ENT], int(cond[len("entropy_only"):]))
    elif cond.startswith("word_repeat"):
        phon = permute_word_types(phon, cols, int(cond[len("word_repeat"):]))
    elif cond.startswith("matched"):
        phon = permute_within_strata(phon, cols, int(cond[len("matched"):]), key_fn=strata_phoneme)
    elif cond.startswith("position_length"):
        phon = permute_within_strata(phon, cols, int(cond[len("position_length"):]))
    Y, ch_names, fs = load_lowfreq(rc, sub) if low_freq() else rc.load_hg_raw(sub)
    n_t = Y.shape[0]
    if plant:
        s = planted_signal(rc, phon_real, n_t, fs)
        Y = Y + plant * Y.std(axis=0, keepdims=True) * s[:, None]
    fams = FAMS[base]
    blocks, pca = {}, {}
    if "mel" in fams:
        blocks["mel"], pca["mel"] = rc.load_mel_raw(n_t), rc.N_MEL_PCS
    if "whisper" in fams:
        wpath = UNTRAINED_WHISPER if base == "whisper_untrained" else WHISPER_FILES.get(base)
        blocks["whisper"], pca["whisper"] = rc.load_whisper_raw(n_t, wpath), rc.N_WHISPER_PCS
    if "word_sparse" in fams:
        blocks["word_sparse"], pca["word_sparse"] = build_event_series(rc.make_word_features(words), n_t, fs), None
    if "gpt2xl_emb" in fams or "gpt2xl_prev" in fams or "gpt2xl_phon" in fams:
        emb, wi = rc.load_gpt_event_per_word_raw(words)
        if "gpt2xl_emb" in fams:
            blocks["gpt2xl_emb"], pca["gpt2xl_emb"] = rc.build_gpt_event_series(words, emb, wi, n_t, fs), rc.N_GPT_PCS
        if "gpt2xl_prev" in fams:
            prev = np.zeros_like(emb)
            prev[1:] = emb[:-1]
            blocks["gpt2xl_prev"], pca["gpt2xl_prev"] = rc.build_gpt_event_series(words, prev, wi, n_t, fs), rc.N_GPT_PCS
        if "gpt2xl_phon" in fams:
            blocks["gpt2xl_phon"], pca["gpt2xl_phon"] = gpt_phoneme_series(words, phon_real, emb, wi, n_t, fs), rc.N_WHISPER_PCS
    if "phon_sparse" in fams:
        Xp = build_event_series(phon_features(rc, phon, with_entropy), n_t, fs)
        if cohort_band() and not with_entropy:
            other = [j for j in range(Xp.shape[1]) if j not in COHORT_COLS]
            blocks["cohort3"], pca["cohort3"] = Xp[:, COHORT_COLS], None
            Xp = Xp[:, other]
        blocks["phon_sparse"], pca["phon_sparse"] = Xp, None
    if "sent" in fams:
        blocks["sent"], pca["sent"] = build_event_series(rc.make_sent_features(words), n_t, fs), None
    for f in blocks:
        if f in EVENT_BLOCKS:
            blocks[f] = boxcar(blocks[f], box_width())
    order = [f for f in ORDER if f in blocks]
    res = fit_clean_banded_ridge({f: blocks[f] for f in order}, {f: pca[f] for f in order}, Y, DEFAULT_LAGS_MS, fs, n_folds=5, n_iter=penalty_settings()[1], log_alpha_hi=penalty_settings()[0])
    cv = res["cv_r_mean"]
    df = pd.DataFrame({"subject": sub, "electrode": ch_names, "cv_r": cv.astype(float)})
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / shard_name(variant, cond, sub, plant), index=False)
    return sub, float(cv.max()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--plant", type=float, default=0.0)
    ap.add_argument("--subs", nargs="*", default=[f"sub-{i:02d}" for i in range(1, 10)])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    todo = [(a.variant, a.cond, s, a.plant) for s in a.subs if not (OUT / shard_name(a.variant, a.cond, s, a.plant)).exists()]
    print(f"[A13] {a.variant} {a.cond} plant={a.plant}: {len(todo)} to fit", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for sub, mx, dt in ex.map(fit_subject, todo):
            print(f"[A13] {a.variant} {a.cond} plant={a.plant} {sub}: max={mx:.3f} {dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
