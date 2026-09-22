from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _segment_folds(T, n_folds):
    edges = np.linspace(0, T, n_folds + 1, dtype=int)
    return [
        (
            np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], T)]),
            np.arange(edges[i], edges[i + 1]),
        )
        for i in range(n_folds)
    ]


def _solve_primal(XtX, Xty, feature_groups, alpha_vec):
    F = XtX.shape[0]
    A = XtX.copy()
    diag_add = np.zeros(F, dtype=A.dtype)
    for g, sl in enumerate(feature_groups):
        diag_add[sl] += alpha_vec[g]
    A[np.arange(F), np.arange(F)] += diag_add
    return np.linalg.solve(A, Xty).astype(np.float32)


def _build_lagged(X, lags_samp):
    T, F = X.shape
    L = len(lags_samp)
    out = np.zeros((T, F * L), dtype=np.float32)
    for j in range(F):
        for li, lag in enumerate(lags_samp):
            col = j * L + li
            if lag == 0:
                out[:, col] = X[:, j]
            elif lag > 0:
                out[lag:, col] = X[:-lag, j]
            else:
                out[:lag, col] = X[-lag:, j]
    return out


def fit_clean_banded_ridge(
    base_blocks: dict[str, np.ndarray],
    pca_config: dict[str, int | None],
    Y: np.ndarray,
    lags_ms: list[int] | tuple[int, ...],
    fs: float,
    n_folds: int = 5,
    n_iter: int = 30,
    log_alpha_lo: float = 0.0,
    log_alpha_hi: float = 5.0,
    val_frac: float = 0.2,
    rng_seed: int = 20260509,
    return_weights: bool = False,
) -> dict:
    """Fold-clean banded ridge.

    Inside each outer fold:
      1. Fit PCA per family on TRAIN slice only; transform train+val+test
      2. Z-score features per column on train; apply to val+test
      3. Z-score Y on train_inner; apply to val+test
      4. Lagged design built from train+val+test
      5. Random search over per-family alpha simplex; pick best per-target
         on val slice
      6. Refit on full train (train_inner + val) with chosen alphas
      7. Score on test
    """
    rng = np.random.default_rng(rng_seed)
    T = next(iter(base_blocks.values())).shape[0]
    n_targets = Y.shape[1]

    assert all(b.shape[0] == T for b in base_blocks.values())

    lags_samp = [int(round(lm * fs / 1000)) for lm in lags_ms]
    n_lags = len(lags_samp)
    families = list(base_blocks.keys())
    n_families = len(families)
    candidate_alphas = 10 ** rng.uniform(log_alpha_lo, log_alpha_hi,
                                          size=(n_iter, n_families))

    cv_r = np.zeros((n_folds, n_targets), dtype=np.float32)
    outer = _segment_folds(T, n_folds)

    weights_per_fold = None

    for fi, (tr_outer, te) in enumerate(outer):
        n_tr = len(tr_outer)
        n_val = int(round(n_tr * val_frac))
        tr_inner = tr_outer[: n_tr - n_val]
        val = tr_outer[n_tr - n_val :]

        proj_blocks_inner = []
        proj_blocks_full_train = []
        proj_blocks_val = []
        proj_blocks_test = []
        family_widths = []
        for fam in families:
            B = base_blocks[fam]
            n_pc = pca_config.get(fam)
            B_tr = B[tr_inner]
            if n_pc is not None and B.shape[1] > n_pc:
                mu = B_tr.mean(axis=0, keepdims=True)
                sd = B_tr.std(axis=0, keepdims=True) + 1e-9
                B_tr_z = (B_tr - mu) / sd
                pca = PCA(n_components=n_pc).fit(B_tr_z)
                B_full_train_z = (B[tr_outer] - mu) / sd
                B_val_z = (B[val] - mu) / sd
                B_te_z = (B[te] - mu) / sd
                proj_inner = pca.transform(B_tr_z).astype(np.float32)
                proj_full_train = pca.transform(B_full_train_z).astype(np.float32)
                proj_val = pca.transform(B_val_z).astype(np.float32)
                proj_test = pca.transform(B_te_z).astype(np.float32)
            else:
                mu = B_tr.mean(axis=0, keepdims=True)
                sd = B_tr.std(axis=0, keepdims=True) + 1e-9
                proj_inner = ((B[tr_inner] - mu) / sd).astype(np.float32)
                proj_full_train = ((B[tr_outer] - mu) / sd).astype(np.float32)
                proj_val = ((B[val] - mu) / sd).astype(np.float32)
                proj_test = ((B[te] - mu) / sd).astype(np.float32)
            proj_blocks_inner.append(proj_inner)
            proj_blocks_full_train.append(proj_full_train)
            proj_blocks_val.append(proj_val)
            proj_blocks_test.append(proj_test)
            family_widths.append(proj_inner.shape[1])

        Xb_inner = np.concatenate(proj_blocks_inner, axis=1)
        Xb_full_train = np.concatenate(proj_blocks_full_train, axis=1)
        Xb_val = np.concatenate(proj_blocks_val, axis=1)
        Xb_test = np.concatenate(proj_blocks_test, axis=1)

        X_inner = _build_lagged(Xb_inner, lags_samp)
        X_val = _build_lagged(Xb_val, lags_samp)
        X_full_train = _build_lagged(Xb_full_train, lags_samp)
        X_test = _build_lagged(Xb_test, lags_samp)

        col_off = 0
        family_groups = []
        for w in family_widths:
            family_groups.append(slice(col_off * n_lags, (col_off + w) * n_lags))
            col_off += w

        mu_y = Y[tr_inner].mean(axis=0, keepdims=True)
        sd_y = Y[tr_inner].std(axis=0, keepdims=True) + 1e-9
        Y_tr_inner_z = (Y[tr_inner] - mu_y) / sd_y
        Y_val_z = (Y[val] - mu_y) / sd_y
        Y_te_z = (Y[te] - mu_y) / sd_y

        XtX_inner = X_inner.T @ X_inner
        Xty_inner = X_inner.T @ Y_tr_inner_z
        scores = np.zeros((n_iter, n_targets), dtype=np.float32)
        for ai in range(n_iter):
            w = _solve_primal(XtX_inner, Xty_inner, family_groups, candidate_alphas[ai])
            Yp_val = X_val @ w
            for j in range(n_targets):
                yp, yt = Yp_val[:, j], Y_val_z[:, j]
                if yp.std() > 0 and yt.std() > 0:
                    scores[ai, j] = float(np.corrcoef(yp, yt)[0, 1])
        best_alpha_idx = scores.argmax(axis=0)

        mu_y_full = Y[tr_outer].mean(axis=0, keepdims=True)
        sd_y_full = Y[tr_outer].std(axis=0, keepdims=True) + 1e-9
        Y_full_train_z = (Y[tr_outer] - mu_y_full) / sd_y_full
        Y_te_zf = (Y[te] - mu_y_full) / sd_y_full
        XtX = X_full_train.T @ X_full_train
        Xty = X_full_train.T @ Y_full_train_z

        Y_pred = np.zeros((len(te), n_targets), dtype=np.float32)
        if return_weights:
            n_design = X_full_train.shape[1]
            if weights_per_fold is None:
                weights_per_fold = np.zeros(
                    (n_folds, n_design, n_targets), dtype=np.float32)
        for ai in np.unique(best_alpha_idx):
            mask_t = best_alpha_idx == ai
            tgt = np.where(mask_t)[0]
            w = _solve_primal(XtX, Xty[:, tgt], family_groups, candidate_alphas[ai])
            Y_pred[:, tgt] = X_test @ w
            if return_weights:
                weights_per_fold[fi, :, tgt] = w.T

        for j in range(n_targets):
            yp, yt = Y_pred[:, j], Y_te_zf[:, j]
            if yp.std() > 0 and yt.std() > 0:
                cv_r[fi, j] = float(np.corrcoef(yp, yt)[0, 1])

    out = {
        "cv_r_per_fold": cv_r,
        "cv_r_mean": cv_r.mean(axis=0),
    }
    if return_weights and weights_per_fold is not None:
        out["weights_per_fold"] = weights_per_fold
        out["weights_mean"] = weights_per_fold.mean(axis=0)
        out["family_widths"] = family_widths
        out["families"] = families
        out["n_lags"] = n_lags
        out["lags_ms"] = list(lags_ms)
    return out
