from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_FS = 100.0
DEFAULT_LAGS_MS = tuple(range(0, 1001, 50))


def load_hg(
    sub: str,
    root: Path = Path("data/ds005574"),
    target_fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, list[str], float]:
    """Return HG envelope (T, n_ch), channel names, and sample rate.

    Z-scores per channel before returning.
    """
    fif = root / "derivatives" / "ecogprep" / sub / "ieeg" / \
        f"{sub}_task-podcast_desc-highgamma_ieeg.fif"
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    if raw.info["sfreq"] != target_fs:
        raw.resample(target_fs, npad="auto", verbose="ERROR")
    data = raw.get_data().T
    med = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - med), axis=0, keepdims=True) + 1e-9
    data = (data - med) / (1.4826 * mad)
    return data.astype(np.float32), list(raw.ch_names), float(raw.info["sfreq"])


@dataclass
class FeatureSpec:
    """One feature column source. `df` has columns word_idx/phoneme_idx
    and `value_col`. `onset_col` indicates the timing column to use."""
    name: str
    df: pd.DataFrame
    value_col: str
    onset_col: str = "onset_sec"

    def values_and_onsets(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.df[self.value_col].to_numpy(dtype=np.float64)
        t = self.df[self.onset_col].to_numpy(dtype=np.float64)
        return v, t


def build_event_series(
    feats: Sequence[FeatureSpec],
    n_samples: int,
    fs: float,
) -> np.ndarray:
    """Return X_event (n_samples, n_features): each column has the
    feature's value at each event-onset time bin, zero elsewhere.
    Each non-zero column is divided by its (non-zero) std so columns
    have comparable scale; constant indicators are preserved as 1s
    (we DON'T center because that would destroy the spike-train
    structure of indicator features).
    """
    X = np.zeros((n_samples, len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        v, t = f.values_and_onsets()
        idx = np.round(t * fs).astype(int)
        keep = (idx >= 0) & (idx < n_samples) & np.isfinite(v)
        idx, v = idx[keep], v[keep]
        if len(v) == 0:
            continue
        sd = float(v.std())
        if sd > 1e-9:
            v = v / sd
        X[idx, j] = v.astype(np.float32)
    return X


def lag_design(
    X_event: np.ndarray, fs: float, lags_ms: Iterable[int]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build the lagged design matrix X (T, F * n_lags) and return the
    list of (feature_idx, lag_idx) for each output column."""
    T, F = X_event.shape
    lags = [int(round(lm * fs / 1000)) for lm in lags_ms]
    L = len(lags)
    X_full = np.zeros((T, F * L), dtype=np.float32)
    cols: list[tuple[int, int]] = []
    for j in range(F):
        for li, lag in enumerate(lags):
            col = j * L + li
            cols.append((j, li))
            if lag == 0:
                X_full[:, col] = X_event[:, j]
            elif lag > 0:
                X_full[lag:, col] = X_event[:-lag, j]
            else:
                X_full[:lag, col] = X_event[-lag:, j]
    return X_full, cols


def segment_kfold(n_samples: int, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous k-fold split (LOSO-CV). Returns list of (train_idx, test_idx)."""
    edges = np.linspace(0, n_samples, k + 1, dtype=int)
    folds = []
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1])
        train = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], n_samples)])
        folds.append((train, test))
    return folds


def fit_banded_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    feature_groups: list[slice],
    alphas: np.ndarray | None = None,
    n_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit himalaya banded ridge: per-feature-group alpha learned from
    a small-grid random search, per-target."""
    from himalaya.kernel_ridge import MultipleKernelRidgeCV
    from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer
    from himalaya.scoring import correlation_score

    Xs = [X[:, sl] for sl in feature_groups]
    kernelizers = [(f"g{i}", Kernelizer(kernel="linear"), feature_groups[i])
                   for i in range(len(feature_groups))]
    cker = ColumnKernelizer(kernelizers)
    from himalaya.backend import set_backend
    set_backend("numpy")
    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver="random_search",
        solver_params={"n_iter": n_iter, "alphas": np.logspace(-2, 4, 10) if alphas is None else alphas},
    )
    K = cker.fit_transform(X)
    model.fit(K, Y)
    return model, cker


def _ridge_per_group_solve(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    feature_groups: list[slice],
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lighter alternative: independently fit ridge per feature group with
    leave-one-fold-out alpha selection, then sum predictions. Returns
    (best_alpha_per_group, weight_matrix).

    NOTE: this is a simpler stand-in for true banded ridge if himalaya
    integration becomes painful. Each group gets its own alpha; predictions
    are summed (equivalent to fitting w with block-diagonal regularization).
    """
    from sklearn.linear_model import RidgeCV
    n_targets = Y_tr.shape[1]
    W = np.zeros((X_tr.shape[1], n_targets), dtype=np.float32)
    best_alphas = np.zeros(len(feature_groups))
    for g, sl in enumerate(feature_groups):
        Xg = X_tr[:, sl]
        m = RidgeCV(alphas=alphas, fit_intercept=True, cv=3)
        m.fit(Xg, Y_tr)
        W[sl, :] = m.coef_.T.astype(np.float32)
        best_alphas[g] = m.alpha_
    return best_alphas, W


def cv_per_electrode(
    X: np.ndarray,
    Y: np.ndarray,
    feature_groups: list[slice] | None = None,
    k: int = 5,
    alphas: np.ndarray | None = None,
) -> dict:
    """Leave-one-segment-out CV with per-electrode RidgeCV over the full
    design matrix (TRF-style). Returns per-electrode mean cv_r,
    per-fold cv_r, and best alpha per electrode (from final-fold).

    `feature_groups` is accepted for API symmetry but ignored — for our
    scale, single-ridge with per-target alpha selection beats summed
    per-group ridges and matches the standard TRF approach. For
    decomposing into unique variance per group, use `unique_variance`
    below which fits separate models with each group dropped.
    """
    if alphas is None:
        alphas = np.logspace(0, 5, 8)
    n_t = Y.shape[1]
    folds = segment_kfold(X.shape[0], k)
    cv_r = np.zeros((k, n_t), dtype=np.float32)
    best_alpha = np.zeros(n_t, dtype=np.float32)
    from sklearn.linear_model import RidgeCV
    for fi, (tr, te) in enumerate(folds):
        mu = Y[tr].mean(axis=0, keepdims=True)
        sd = Y[tr].std(axis=0, keepdims=True) + 1e-9
        Y_tr = (Y[tr] - mu) / sd
        Y_te = (Y[te] - mu) / sd
        m = RidgeCV(alphas=alphas, fit_intercept=True, alpha_per_target=True)
        m.fit(X[tr], Y_tr)
        Yp = m.predict(X[te]).astype(np.float32)
        for ei in range(n_t):
            cv_r[fi, ei] = _pearson(Yp[:, ei], Y_te[:, ei])
        if fi == k - 1:
            best_alpha = m.alpha_.astype(np.float32)
    return {
        "cv_r_per_fold": cv_r,
        "cv_r_mean": cv_r.mean(axis=0),
        "cv_r_se": cv_r.std(axis=0) / np.sqrt(k),
        "best_alpha": best_alpha,
    }


def unique_variance(
    X: np.ndarray,
    Y: np.ndarray,
    feature_groups: dict[str, slice],
    k: int = 5,
    alphas: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute unique variance contribution per group via leave-group-out.

    For each group g:
        unique_r[g] = cv_r(joint) - cv_r(joint without group g)
    Returns dict: group_name -> per-electrode unique cv_r contribution.
    Also returns 'joint' = cv_r of full model.
    """
    out: dict[str, np.ndarray] = {}
    full = cv_per_electrode(X, Y, k=k, alphas=alphas)
    out["joint"] = full["cv_r_mean"]
    all_cols = np.arange(X.shape[1])
    for name, sl in feature_groups.items():
        mask = np.ones(X.shape[1], dtype=bool)
        mask[sl] = False
        Xred = X[:, mask]
        red = cv_per_electrode(Xred, Y, k=k, alphas=alphas)
        out[name] = full["cv_r_mean"] - red["cv_r_mean"]
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
