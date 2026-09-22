import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import whisper_past_only_features as p28


@pytest.fixture(scope="module")
def audio():
    return p28.load_audio()


@pytest.fixture(scope="module")
def ex():
    return p28.Extractor("fp16")


def test_window_is_blind_to_future(audio):
    rng = np.random.default_rng(0)
    for k in (0, 10, 1499, 1500, 40000, p28.N_FRAMES - 1):
        future_noise = audio.copy()
        cut = (k + 1) * p28.HOP
        future_noise[cut:] = rng.standard_normal(future_noise.size - cut).astype(np.float32)
        w0, w1 = p28.window(p28.pad_audio(audio), k), p28.window(p28.pad_audio(future_noise), k)
        assert w0.shape == (p28.WIN,) and np.array_equal(w0, w1)
        past_noise = audio.copy()
        past_noise[max(0, cut - 160):cut] += 1.0
        assert not np.array_equal(w0, p28.window(p28.pad_audio(past_noise), k))


def test_window_ends_at_frame_end(audio):
    k = 2000
    w = p28.window(p28.pad_audio(audio), k)
    assert np.array_equal(w, audio[(k + 1) * p28.HOP - p28.WIN:(k + 1) * p28.HOP])
    w_early = p28.window(p28.pad_audio(audio), 99)
    assert np.all(w_early[: p28.WIN - 100 * p28.HOP] == 0) and np.array_equal(w_early[p28.WIN - 100 * p28.HOP:], audio[: 100 * p28.HOP])


def test_gpu_path_matches_released_features(audio, ex):
    """Frame 1499 of the first released chunk saw the same 30 s of audio as our window k=1499 (no future exists inside that chunk)."""
    import torch
    with h5py.File(common.MAIN / "data" / "ds005574" / "stimuli" / "whisper-medium" / "encoder.hdf5", "r") as f:
        released = f["vectors"][:, :1500].T
    with torch.no_grad():
        feats = ex.fe([audio[: p28.WIN]], sampling_rate=p28.SR, return_tensors="pt", device=ex.device)["input_features"]
        h = ex.enc(feats.to(ex.device, ex.dtype)).last_hidden_state[0].float().cpu().numpy()
    assert np.corrcoef(h.ravel(), released.ravel())[0, 1] > 0.9995
    wc, la = ex([p28.window(p28.pad_audio(audio), 1499)])
    assert np.corrcoef(wc[0], released[1499])[0, 1] > 0.999
    assert np.corrcoef(la[0], released[1499 - p28.LA])[0, 1] > 0.999


def test_output_layout_matches_loader():
    f = common.OUT / "features" / "whisper_causal.hdf5"
    if not f.exists():
        pytest.skip("extraction not run yet")
    for name in ("whisper_causal.hdf5", "whisper_causal_la500.hdf5"):
        with h5py.File(common.OUT / "features" / name, "r") as h:
            assert h["vectors"].shape == (1024, 90000)
            v = h["vectors"][:, ::997]
            assert np.isfinite(v).all() and v.std() > 0
