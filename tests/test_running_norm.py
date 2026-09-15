import numpy as np

from rl.core.running_norm import RunningMeanStd


def test_single_batch_matches_numpy_mean_and_var():
    rms = RunningMeanStd(dim=3)
    x = np.array([[1.0, 10.0, -5.0], [3.0, 20.0, -1.0], [5.0, 30.0, 3.0]], dtype=np.float32)
    rms.update(x)
    assert np.allclose(rms.mean, x.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.var, x.var(axis=0), atol=1e-3)


def test_two_sequential_batches_match_combined_numpy_stats():
    # Welford's batched update must equal the one-shot stats over both
    # batches concatenated — this is the non-obvious part (running stats
    # updated incrementally across many update() calls, never all at once).
    rms = RunningMeanStd(dim=2)
    batch1 = np.array([[1.0, 100.0], [2.0, 200.0]], dtype=np.float32)
    batch2 = np.array([[3.0, 300.0], [4.0, 400.0], [5.0, 500.0]], dtype=np.float32)
    rms.update(batch1)
    rms.update(batch2)
    combined = np.concatenate([batch1, batch2], axis=0)
    assert np.allclose(rms.mean, combined.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.var, combined.var(axis=0), atol=1e-3)


def test_normalize_scales_by_std_without_centering():
    # A term whose running mean is far from zero must NOT get shifted toward
    # zero by normalize() — only scaled — so a bounded [0,1] term like
    # clearance_reward keeps its 0-boundary meaning.
    rms = RunningMeanStd(dim=1)
    rms.update(np.array([[10.0], [10.0], [10.0], [10.0]], dtype=np.float32))
    out = rms.normalize(np.array([[10.0]], dtype=np.float32))
    # var is ~0 here so std is tiny; feed a case with real spread instead.
    rms2 = RunningMeanStd(dim=1)
    rms2.update(np.array([[8.0], [10.0], [12.0]], dtype=np.float32))  # mean=10, var=8/3
    std = np.sqrt(rms2.var[0] + 1e-8)
    out2 = rms2.normalize(np.array([[10.0]], dtype=np.float32))[0, 0]
    assert np.isclose(out2, 10.0 / std, atol=1e-3)  # not (10-mean)/std == 0


def test_normalize_clips_outliers():
    rms = RunningMeanStd(dim=1)
    rms.update(np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32))
    out = rms.normalize(np.array([[1000.0]], dtype=np.float32), clip=10.0)
    assert out[0, 0] == 10.0


def test_state_dict_round_trip_preserves_stats():
    rms = RunningMeanStd(dim=2)
    rms.update(np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32))
    state = rms.state_dict()

    restored = RunningMeanStd(dim=2)
    restored.load_state_dict(state)
    assert np.allclose(restored.mean, rms.mean)
    assert np.allclose(restored.var, rms.var)
    assert restored.count == rms.count
