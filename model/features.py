from __future__ import annotations

import os

from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from model.banded_ridge import fit_clean_banded_ridge
from model.encoding import (DEFAULT_LAGS_MS, FeatureSpec,
                                    build_event_series)

ROOT = Path(os.environ.get("PODCAST_ROOT", Path(__file__).resolve().parent.parent))
SUBS = [f"sub-{i:02d}" for i in range(1, 10)]
TARGET_FS = 100.0
N_MEL_PCS = 16
N_WHISPER_PCS = 64
N_GPT_PCS = 32


def load_hg_raw(sub: str, target_fs: float = TARGET_FS):
    fif = (ROOT / "data" / "ds005574" / "derivatives" / "ecogprep" / sub
            / "ieeg" / f"{sub}_task-podcast_desc-highgamma_ieeg.fif")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if raw.info["sfreq"] != target_fs:
        raw.resample(target_fs, npad="auto", verbose="ERROR")
    return (raw.get_data().T.astype(np.float32),
            list(raw.ch_names), float(raw.info["sfreq"]))


def load_mel_raw(n_t: int) -> np.ndarray:
    with h5py.File(ROOT / "data" / "ds005574" / "stimuli" / "spectral"
                    / "spectrogram.hdf5", "r") as f:
        mel = f["vectors"][:].T
    if mel.shape[0] > n_t:
        mel = mel[:n_t]
    elif mel.shape[0] < n_t:
        mel = np.pad(mel, ((0, n_t - mel.shape[0]), (0, 0)))
    return mel.astype(np.float32)


def load_whisper_raw(n_t: int,
                       hdf5_path: Path | None = None) -> np.ndarray:
    p = hdf5_path or (ROOT / "data" / "ds005574" / "stimuli"
                       / "whisper-medium" / "encoder.hdf5")
    with h5py.File(p, "r") as f:
        w = f["vectors"][:].T
    t_w = np.arange(w.shape[0]) * 2
    t_target = np.arange(n_t)
    out = np.zeros((n_t, w.shape[1]), dtype=np.float32)
    for j in range(w.shape[1]):
        out[:, j] = np.interp(t_target, t_w, w[:, j])
    return out


def load_gpt_event_per_word_raw(words: pd.DataFrame):
    pt = pd.read_csv(ROOT / "data" / "ds005574" / "stimuli" / "gpt2-xl"
                      / "transcript.tsv", sep="\t")
    with h5py.File(ROOT / "data" / "ds005574" / "stimuli" / "gpt2-xl"
                    / "features.hdf5", "r") as f:
        emb = f["layer-24"][:]
    df = pd.DataFrame(emb.astype(np.float32))
    df["word_idx"] = pt["word_idx"].values
    pooled = df.groupby("word_idx", sort=True).mean().reset_index()
    word_idxs = pooled["word_idx"].astype(int).to_numpy()
    embmat = pooled.drop(columns=["word_idx"]).to_numpy().astype(np.float32)
    return embmat, word_idxs


def build_gpt_event_series(words, gpt_embmat, gpt_word_idxs, n_t, fs):
    n_dim = gpt_embmat.shape[1]
    out = np.zeros((n_t, n_dim), dtype=np.float32)
    word_to_emb = {int(wi): gpt_embmat[i] for i, wi in enumerate(gpt_word_idxs)}
    onsets = words["onset_sec"].to_numpy()
    word_ids = words["word_idx"].to_numpy()
    for w_id, t in zip(word_ids, onsets):
        wi = int(w_id)
        if wi in word_to_emb:
            idx = int(round(t * fs))
            if 0 <= idx < n_t:
                out[idx] = word_to_emb[wi]
    return out


def make_word_features(words: pd.DataFrame) -> list[FeatureSpec]:
    return [
        FeatureSpec("word_onset",  words.assign(_=1.0), "_"),
        FeatureSpec("surprisal",   words, "surprisal_gpt2xl"),
        FeatureSpec("entropy",     words, "entropy_gpt2xl"),
        FeatureSpec("freq_lg10",   words, "freq_subtlex_lg10"),
        FeatureSpec("is_content",
                     words.assign(_c=words.is_content_word.astype(float)), "_c"),
        FeatureSpec("word_dur",    words, "word_duration_sec"),
        FeatureSpec("pos_in_sent", words, "position_in_sentence_norm"),
    ]


def make_phon_features(phon: pd.DataFrame) -> list[FeatureSpec]:
    return [
        FeatureSpec("phon_onset", phon.assign(_=1.0), "_", "phoneme_onset_sec"),
        FeatureSpec("cohort_surp", phon, "phoneme_surprisal_in_cohort",
                     "phoneme_onset_sec"),
        FeatureSpec("log_cohort",
                     phon.assign(_l=np.log10(phon.cohort_size_before
                                              .fillna(1).clip(lower=1))), "_l",
                     "phoneme_onset_sec"),
        FeatureSpec("dist_from_UP", phon, "distance_from_uniqueness_point",
                     "phoneme_onset_sec"),
        FeatureSpec("pos_in_word", phon, "position_in_word_norm",
                     "phoneme_onset_sec"),
        FeatureSpec("is_initial",
                     phon.assign(_=phon.is_word_initial.astype(float)), "_",
                     "phoneme_onset_sec"),
        FeatureSpec("is_final",
                     phon.assign(_=phon.is_word_final.astype(float)), "_",
                     "phoneme_onset_sec"),
        FeatureSpec("is_vowel",
                     phon.assign(_=(phon.manner == "vowel").astype(float)),
                     "_", "phoneme_onset_sec"),
        FeatureSpec("is_plosive",
                     phon.assign(_=(phon.manner == "plosive").astype(float)),
                     "_", "phoneme_onset_sec"),
        FeatureSpec("is_fricative",
                     phon.assign(_=(phon.manner == "fricative").astype(float)),
                     "_", "phoneme_onset_sec"),
        FeatureSpec("is_nasal",
                     phon.assign(_=(phon.manner == "nasal").astype(float)),
                     "_", "phoneme_onset_sec"),
    ]


def make_sent_features(words: pd.DataFrame) -> list[FeatureSpec]:
    return [
        FeatureSpec("sent_initial",
                     words.assign(_=words.is_sentence_initial.astype(float)), "_"),
        FeatureSpec("sent_final",
                     words.assign(_=words.is_sentence_final.astype(float)), "_"),
        FeatureSpec("pos_in_sent_q3", words, "position_in_sentence_norm"),
        FeatureSpec("node_count", words, "node_count_close"),
    ]


def load_features():
    words = pd.read_parquet(ROOT / "data" / "features" / "words.parquet")
    phon = pd.read_parquet(ROOT / "data" / "features"
                            / "phonemes_hybrid.parquet")
    phon = phon.dropna(subset=["phoneme_surprisal_in_cohort"]).reset_index(drop=True)
    return words, phon


def run_variant(variant_name: str,
                  family_selector,
                  out_path: str | Path,
                  whisper_hdf5: Path | None = None,
                  subs: list[str] = SUBS):
    """Fit `fit_clean_banded_ridge` per subject under a feature-subset
    selector. Saves per-electrode cv_r to `out_path`.

    family_selector receives the full six-family base_blocks + pca_config
    and returns a tuple (subset_blocks, subset_pca_config) restricting to
    the families this variant uses.
    """
    words, phon = load_features()
    q1 = make_word_features(words)
    q2 = make_phon_features(phon)
    q3 = make_sent_features(words)
    el = pd.read_parquet(ROOT / "data" / "derived" / "electrodes.parquet")

    rows = []
    for sub in subs:
        print(f"\n[{variant_name}] === {sub} ===", flush=True)
        Y, ch_names, fs = load_hg_raw(sub)
        n_t = Y.shape[0]
        print(f"  HG: T={n_t}, n_ch={Y.shape[1]}, fs={fs}", flush=True)

        mel_raw = load_mel_raw(n_t)
        whisper_raw = load_whisper_raw(n_t, hdf5_path=whisper_hdf5)
        gpt_embmat, gpt_word_idxs = load_gpt_event_per_word_raw(words)
        gpt_event_raw = build_gpt_event_series(words, gpt_embmat,
                                                  gpt_word_idxs, n_t, fs)
        word_event = build_event_series(q1, n_t, fs)
        phon_event = build_event_series(q2, n_t, fs)
        sent_event = build_event_series(q3, n_t, fs)

        full_blocks = {
            "mel":         mel_raw,
            "whisper":     whisper_raw,
            "word_sparse": word_event,
            "gpt2xl_emb":  gpt_event_raw,
            "phon_sparse": phon_event,
            "sent":        sent_event,
        }
        full_pca = {
            "mel":         N_MEL_PCS,
            "whisper":     N_WHISPER_PCS,
            "word_sparse": None,
            "gpt2xl_emb":  N_GPT_PCS,
            "phon_sparse": None,
            "sent":        None,
        }
        blocks, pca = family_selector(full_blocks, full_pca)
        print(f"  variant families: {list(blocks.keys())}", flush=True)

        result = fit_clean_banded_ridge(
            blocks, pca, Y, DEFAULT_LAGS_MS, fs,
            n_folds=5, n_iter=30,
        )
        cv_r = result["cv_r_mean"]
        max_r = float(cv_r.max())
        med_strong = float(np.median(cv_r[cv_r > 0.05])) if (cv_r > 0.05).any() else float("nan")
        print(f"  [{variant_name}] cv_r: max={max_r:.3f}, "
               f"median(>0.05)={med_strong:.3f}, "
               f"n>0.1={int((cv_r>0.1).sum())}, n>0.3={int((cv_r>0.3).sum())}",
               flush=True)

        el_sub = el[el.subject == sub].set_index("name")
        for ei, ch in enumerate(ch_names):
            meta = el_sub.loc[ch].to_dict() if ch in el_sub.index else {}
            rows.append({
                "subject": sub, "electrode": ch,
                "cv_r": float(cv_r[ei]),
                "ho_label": meta.get("ho_label"),
                "is_lang_roi": bool(meta.get("is_lang_roi") or False),
                "is_stg_or_mtg": bool(meta.get("is_stg_or_mtg") or False),
                "x": meta.get("x"), "y": meta.get("y"), "z": meta.get("z"),
                "hemisphere": meta.get("hemisphere"),
            })

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[{variant_name}] saved: {out_path}", flush=True)
    print(f"[{variant_name}] summary: max={df.cv_r.max():.3f}, "
           f"median(>0.05)={df[df.cv_r>0.05].cv_r.median():.3f}, "
           f"n>0.1={int((df.cv_r>0.1).sum())}, n>0.3={int((df.cv_r>0.3).sum())}",
           flush=True)
    return df
